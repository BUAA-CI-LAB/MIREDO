import torch
import torch.nn as nn

class testSingleNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # self.func = update_zeroPoint_someLayers
        self.func = lambda x:x
        self.block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),           # resnet18-layer1
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),          # resnet18-layer3
            ) 

    def forward(self, x):
        x = self.block(x)
        return x