import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *

class MixedOp(nn.Module):
    def __init__(self,C,stride):
        super(MixedOp,self).__init__()

        self.ops = nn.ModuleList([SepConv(C,C,3,stride,1),
                                SepConv(C,C,5,stride,2),
                                DilConv(C, C, 3, stride, 2, 2),
                                DilConv(C, C, 5, stride, 4, 2),
                                nn.MaxPool2d(3, stride=stride, padding=1),
                                nn.AvgPool2d(3, stride=stride, padding=1),
                                Identity() if stride == 1 else FactorizedReduce(C, C),
                                Zero(stride)])
        
    def forward(self,x,arch):
        probs = F.softmax(arch,dim=0)
        # print(arch.size())
        output = 0
        for op,prob in zip(self.ops,probs):
            output = output + op(x)*prob
        return output

class NormalCell(nn.Module):
    def __init__(self,C_pp,C_p,C,prev_reduction=False):
        super(NormalCell,self).__init__()
        self.C_out = C

        partial_C = self.C_out//4
        
        if prev_reduction:
            self.preproc_pp = FactorizedReduce(C_pp,partial_C)
        else:
            self.preproc_pp = ReLUConvBN(C_pp,partial_C,1,1,0)
        self.preproc_p = ReLUConvBN(C_p,partial_C,1,1,0)

        self.ops = nn.ModuleList([nn.ModuleList([MixedOp(partial_C,1) for j in range(i+2)]) for i in range(4)])

    def forward(self,x_pp,x_p,arch_param):
        x_pp = self.preproc_pp(x_pp)
        x_p = self.preproc_p(x_p)

        fmap = [x_pp,x_p]

        for i in range(len(self.ops)):
            this_tensor = 0
            for op,f,arch in zip(self.ops[i],fmap,arch_param[i]):
                this_tensor = this_tensor + op(f,arch)
            fmap.append(this_tensor)

        return torch.cat(fmap[2:],dim=1)

class ReductionCell(nn.Module):
    def __init__(self,C_pp,C_p,C):
        super(ReductionCell,self).__init__()
        self.C_out = C

        partial_C = self.C_out//4
        
        self.preproc_pp = ReLUConvBN(C_pp,partial_C,1,1,0)
        self.preproc_p = ReLUConvBN(C_p,partial_C,1,1,0)

        self.ops = nn.ModuleList()
        for i in range(4):
            self.ops.append(nn.ModuleList())
            for j in range(i+2):
                if j<2:
                    self.ops[i].append(MixedOp(partial_C,2))
                else:
                    self.ops[i].append(MixedOp(partial_C,1))
                
        # print(self.ops)

    def forward(self,x_pp,x_p,arch_param):
        x_pp = self.preproc_pp(x_pp)
        x_p = self.preproc_p(x_p)

        fmap = [x_pp,x_p]

        for i in range(len(self.ops)):
            this_tensor = 0
            for op,f,arch in zip(self.ops[i],fmap,arch_param[i]):
                this_tensor = this_tensor + op(f,arch)
            fmap.append(this_tensor)

        return torch.cat(fmap[2:],dim=1)

