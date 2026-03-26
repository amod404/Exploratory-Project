# models/base_net.py
import torch
import torch.nn as nn
from .cells import Cell, ReductionCell

class BaseNet(nn.Module):
    """
    The macro-skeleton of the neural network. 
    It stacks Normal and Reduction cells to process CIFAR-10 images.
    """
    def __init__(self, num_classes=10, init_channels=16, num_cells_per_stage=1):
        super(BaseNet, self).__init__()
        
        # 1. The Stem: Initial processing of the raw 3-channel RGB image
        self.stem = nn.Sequential(
            nn.Conv2d(3, init_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. The Body: Stacking cells dynamically
        self.cells = nn.ModuleList()
        C_curr = init_channels
        
        # We create 3 "stages" (standard for CIFAR-10). 
        # Each stage ends with a ReductionCell to shrink the image.
        for stage in range(3):
            # Add Normal Cells (maintains resolution and channel count)
            for _ in range(num_cells_per_stage):
                self.cells.append(Cell(C_curr, C_curr))
            
            # Add a Reduction Cell (halves resolution, doubles channels)
            # We don't reduce after the very last stage
            if stage < 2:
                C_next = C_curr * 2
                self.cells.append(ReductionCell(C_curr, C_next))
                C_curr = C_next
                
        # 3. The Head: Final classification
        # Flattens any remaining spatial dimensions down to 1x1
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # Outputs the probabilities for the 10 CIFAR classes
        self.classifier = nn.Linear(C_curr, num_classes)

    def forward(self, x):
        """Defines how the data flows through the network."""
        # Pass image through the stem
        out = self.stem(x)
        
        # Pass data through the evolutionary cells
        for cell in self.cells:
            out = cell(out)
            
        # Pool the features and classify
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1) # Flatten the tensor
        out = self.classifier(out)
        
        return out