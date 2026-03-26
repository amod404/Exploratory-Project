# models/blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Standard Convolutional Block with BatchNorm and ReLU."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ConvBlock, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)

class ResBlock(nn.Module):
    """Basic Residual Block. Great starting point for Net2Deeper morphisms."""
    def __init__(self, C_in, C_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(C_in, C_out, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBlock(C_out, C_out, kernel_size=3, stride=1, padding=1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or C_in != C_out:
            # Projection shortcut to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(C_out)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        return F.relu(out)

class Identity(nn.Module):
    """Identity mapping, useful for skip connections."""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    """Zero operation, effectively removing a connection in a DAG."""
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)