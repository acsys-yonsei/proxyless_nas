import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *

class MixedOp(nn.Module):
    def __init__(self,C_in,C_out,stride,ops):
        super(MixedOp,self).__init__()

        self.ops = nn.ModuleList()
        for op in ops:
            self.ops.append(OPS[op](C_in,C_out,stride))

        self.AP_path_alpha = nn.Parameter(torch.Tensor(len(ops)))
        self.AP_path_wb = nn.Parameter(torch.Tensor(len(ops)))

        self.active_index = [0]
        self.inactive_index = None
        
    def forward(self,x):
        def forward_func(cand_ops,active):
            def forward(_x):
                return cand_ops[active](_x)
            return forward
        def backward_func(cand_ops,active,binary_gates):
            def backward(_x,_output,grad_output):
                binary_grads = torch.zeros_like(binary_gates.data)
                with torch.no_grad():
                    for i,op in enumerate(cand_ops):
                        if i != active:
                            out_i = op(_x.data)
                        else:
                            out_i = _output.data
                        grad_i = torch.sum(out_i*grad_output)
                        binary_grads[i] = grad_i
                return binary_grads
            return backward
        
        output = ArchGrad.apply(
            x, self.AP_path_wb, forward_func(self.ops,self.active_index[0]),
            backward_func(self.ops,self.active_index[0],self.AP_path_wb)
        )

        return output

    def binarize(self):
        prob = F.softmax(self.AP_path_alpha,dim=0)
        self.AP_path_wb.data.zero_()

        sample = torch.multinomial(prob.data,1)[0].item()
        self.active_index = [sample]
        self.inactive_index = [_i for _i in range(sample)] + [_i for _i in range(sample+1,len(self.ops))]
        self.AP_path_wb.data[sample] = 1.0

        # regularization?

    def delta(self,i,j):
        if i == j:
            return 1
        else:
            return 0

    def set_arch_grad(self):
        binary_grads = self.AP_path_wb.grad.data
        if isinstance(self.ops[self.active_index[0]],Zero):
            self.AP_path_alpha.grad = None
            return
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
        
        probs = F.softmax(self.AP_path_alpha,dim=0)
        for i in range(len(self.ops)):
            for j in range(len(self.ops)):
                self.AP_path_alpha.grad.data[i] += binary_grads[j]*probs[j]*(self.delta(i,j)-probs[i])
        

class Cell(nn.Module):
    def __init__(self,C_in,C_out,stride):
        super(Cell,self).__init__()
        self.C_out = C_out
        self.C_in = C_in

        conv_candidates = [
            '3x3_MBConv3', '3x3_MBConv6',
            '5x5_MBConv3', '5x5_MBConv6',
            '7x7_MBConv3', '7x7_MBConv6',
        ]

        if stride == 2:
            self.shortcut = False
            modified_conv_candidates = conv_candidates
        else:
            self.shortcut = True
            modified_conv_candidates = conv_candidates + ['Zero']

        self.ops = MixedOp(C_in,C_out,stride,modified_conv_candidates)


    def forward(self,x):
        if self.shortcut:
            skip = x
        else:
            skip = 0

        return self.ops(x)+skip

    def binarize(self):
        self.ops.binarize()

    def set_arch_grad(self):
        self.ops.set_arch_grad()

class ArchGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,binary_gates,forward_func,backward_func):
        ctx.forward_func = forward_func
        ctx.backward_func = backward_func

        detached_x = x.detach()
        detached_x.requires_grad = x.requires_grad
        with torch.enable_grad():
            output = forward_func(detached_x)
        ctx.save_for_backward(detached_x,output)
        return output.data

    @staticmethod
    def backward(ctx,grad_output):
        detached_x, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(output,detached_x,grad_output,only_inputs=True)

        binary_grads = ctx.backward_func(detached_x.data,output.data,grad_output.data)

        return grad_x[0],binary_grads,None,None
