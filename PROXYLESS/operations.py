import torch
import torch.nn as nn

OPS = {
    'Identity': lambda in_C, out_C, S: Identity(),
    'Zero': lambda in_C, out_C, S: Zero(stride=S),
    '3x3_MBConv3': lambda in_C, out_C, S: MBConv(in_C, out_C, 3, S, 3),
    '3x3_MBConv6': lambda in_C, out_C, S: MBConv(in_C, out_C, 3, S, 6),
    #######################################################################################
    '5x5_MBConv3': lambda in_C, out_C, S: MBConv(in_C, out_C, 5, S, 3),
    '5x5_MBConv6': lambda in_C, out_C, S: MBConv(in_C, out_C, 5, S, 6),
    #######################################################################################
    '7x7_MBConv3': lambda in_C, out_C, S: MBConv(in_C, out_C, 7, S, 3),
    '7x7_MBConv6': lambda in_C, out_C, S: MBConv(in_C, out_C, 7, S, 6),
}

class MBConv(nn.Module):
    def __init__(self,C_in,C_out,kernel_size,stride,expand):
        super(MBConv,self).__init__()

        hidden_dim = int(C_in*expand)
        padding = kernel_size//2

        self.op = nn.Sequential(
            nn.Conv2d(C_in,hidden_dim,1,1,0,bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),
            nn.Conv2d(hidden_dim,hidden_dim,kernel_size,stride,padding,groups=hidden_dim,bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),
            nn.Conv2d(hidden_dim,C_out,1,1,0,bias=False),
            nn.BatchNorm2d(C_out)

        )
    
    def forward(self,x):
        return self.op(x)



class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)

