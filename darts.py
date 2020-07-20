import torch.nn as nn
import torch.nn.functional as F
from cell import *

class Darts(nn.Module):
    def __init__(self,C,num_classes,num_layers):
        super(Darts, self).__init__()

        OPS = [
            'none',
            'max_pool_3x3',
            'avg_pool_3x3',
            'skip_connect',
            'sep_conv_3x3',
            'sep_conv_5x5',
            'dil_conv_3x3',
            'dil_conv_5x5'
        ]

        normal_param = [nn.Parameter(torch.Tensor(i+2,8)) for i in range(4)]
        reduc_param = [nn.Parameter(torch.Tensor(i+2,8)) for i in range(4)]
        self.arch_param_normal = nn.ParameterList(normal_param)
        self.arch_param_reduc = nn.ParameterList(reduc_param)

        # print(self.arch_param_normal)
        multi = 3
        C_curr = 3*C

        self.extractor = nn.Sequential(
            nn.Conv2d(3,C_curr,3,padding=1,bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_pp,C_p,C_curr = C_curr,C_curr,C
        self.layers = nn.ModuleList()
        reduction = False
        for i in range(num_layers):
            if i == num_layers//3 or i == 2*num_layers//3:
                C_curr*=2
                reduction = True
                layer = ReductionCell(C_pp,C_p,C_curr,OPS)
            else:
                layer = NormalCell(C_pp,C_p,C_curr,OPS,reduction)
                reduction = False
            self.layers.append(layer)
            C_pp,C_p = C_p,C_curr
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_p, num_classes)

    def forward(self, x):
        x = self.extractor(x)

        x_pp, x_p = x,x
        for layer in self.layers:
            if isinstance(layer,NormalCell):
                x = layer(x_pp,x_p,self.arch_param_normal)
            else:
                x = layer(x_pp,x_p,self.arch_param_reduc)
            x_pp,x_p = x_p,x

        x = self.gap(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        
        return x