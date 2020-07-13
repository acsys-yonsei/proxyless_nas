import torch.nn as nn
import torch.nn.functional as F
from cell import *

class Not_darts_yet(nn.Module):
    def __init__(self,C,num_classes,num_layers):
        super(Not_darts_yet, self).__init__()
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
                layer = ReductionCell(C_pp,C_p,C_curr)
            else:
                layer = NormalCell(C_pp,C_p,C_curr,reduction)
                reduction = False
            self.layers.append(layer)
            C_pp,C_p = C_p,C_curr
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_p, num_classes)

    def forward(self, x):
        x = self.extractor(x)

        x_pp, x_p = x,x
        for layer in self.layers:
            x = layer(x_pp,x_p)
            x_pp,x_p = x_p,x

        x = self.gap(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        
        return x