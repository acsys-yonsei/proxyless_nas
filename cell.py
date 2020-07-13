import torch
import torch.nn as nn
from operations import *

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

        ##########
        self.ops = nn.ModuleList([nn.ModuleList([SepConv(partial_C,partial_C,3,1,1),SepConv(partial_C,partial_C,3,1,1)]),
        nn.ModuleList([SepConv(partial_C,partial_C,3,1,1),SepConv(partial_C,partial_C,3,1,1),None]),
        nn.ModuleList([Identity(),SepConv(partial_C,partial_C,3,1,1),None,None]),
        nn.ModuleList([Identity(),None,DilConv(partial_C, partial_C, 3, 1, 2, 2),None,None])])
        ##########

    def forward(self,x_pp,x_p):
        x_pp = self.preproc_pp(x_pp)
        x_p = self.preproc_p(x_p)

        fmap = [x_pp,x_p]

        for i in range(len(self.ops)):
            this_tensor = 0
            for op,f in zip(self.ops[i],fmap):
                if op != None:
                    this_tensor = this_tensor + op(f)
            fmap.append(this_tensor)

        return torch.cat(fmap[2:],dim=1)

class ReductionCell(nn.Module):
    def __init__(self,C_pp,C_p,C):
        super(ReductionCell,self).__init__()
        self.C_out = C

        partial_C = self.C_out//4
        
        self.preproc_pp = ReLUConvBN(C_pp,partial_C,1,1,0)
        self.preproc_p = ReLUConvBN(C_p,partial_C,1,1,0)

        ##########
        self.ops = nn.ModuleList([nn.ModuleList([nn.MaxPool2d(3, stride=2, padding=1),nn.MaxPool2d(3, stride=2, padding=1)]),
        nn.ModuleList([None,nn.MaxPool2d(3, stride=2, padding=1),Identity()]),
        nn.ModuleList([nn.MaxPool2d(3, stride=2, padding=1),None,Identity(),None]),
        nn.ModuleList([None,nn.MaxPool2d(3, stride=2, padding=1),Identity(),None,None])])
        ##########

    def forward(self,x_pp,x_p):
        x_pp = self.preproc_pp(x_pp)
        x_p = self.preproc_p(x_p)

        fmap = [x_pp,x_p]

        for i in range(len(self.ops)):
            this_tensor = 0
            for op,f in zip(self.ops[i],fmap):
                if op != None:
                    this_tensor = this_tensor + op(f)
            fmap.append(this_tensor)

        return torch.cat(fmap[2:],dim=1)

