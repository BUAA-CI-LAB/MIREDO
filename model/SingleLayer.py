import torch
import torch.nn as nn

class testSingleNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.func = lambda x:x
        self.block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            )

    def forward(self, x):
        x = self.block(x)
        return x
