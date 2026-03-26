# models/cells.py
import torch
import torch.nn as nn
from .blocks import ConvBlock, ResBlock, Identity

class Cell(nn.Module):
    """
    A foundational container for operations. 
    Can be expanded dynamically by the network morphism engine.
    """
    def __init__(self, C_in, C_out, stride=1):
        super(Cell, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        
        # Preprocessing to ensure channel dimensions match before operations
        if C_in != C_out or stride != 1:
            self.preprocess = ConvBlock(C_in, C_out, kernel_size=1, stride=stride, padding=0)
        else:
            self.preprocess = Identity()
            
        # The primary operation block (Starts as a ResBlock, can mutate later)
        self.op = ResBlock(C_out, C_out, stride=1)

    def forward(self, x):
        x = self.preprocess(x)
        return self.op(x)

class ReductionCell(nn.Module):
    """Specifically forces a reduction in spatial dimensions (stride=2)."""
    def __init__(self, C_in, C_out):
        super(ReductionCell, self).__init__()
        # Strided convolution to reduce spatial dimensions (e.g., 32x32 -> 16x16 for CIFAR-10)
        self.reduce = ConvBlock(C_in, C_out, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        return self.reduce(x)