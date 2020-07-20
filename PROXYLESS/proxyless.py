import torch.nn as nn
import torch.nn.functional as F
from cell import *
from operations import *

class Proxyless(nn.Module):
    def __init__(self,C,num_classes,blocks,num_layers):
        super(Proxyless, self).__init__()

        OPS = [
            '3x3_MBConv3', '3x3_MBConv6',
            '5x5_MBConv3', '5x5_MBConv6',
            '7x7_MBConv3', '7x7_MBConv6',
        ]

        self.extractor = nn.Sequential(
            nn.Conv2d(3,8,3,padding=1,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU6()
        )
        C = 8

        self.layers = nn.ModuleList()
        for channel in blocks:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(Cell(C,channel,2))
                    C = channel
                else:
                    self.layers.append(Cell(C,C,1))
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C, num_classes)

    def forward(self, x):
        x = self.extractor(x)

        x_pp, x_p = x,x
        for layer in self.layers:
            x = layer(x)

        x = self.gap(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        
        return x

    def binarize(self):
        for layer in self.layers:
            layer.binarize()
    
    def set_arch_grad(self):
        for layer in self.layers:
            layer.set_arch_grad()