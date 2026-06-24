import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from torchvision.models import alexnet as alex_ori

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, dropout = 0.5, **kwargs):
        super().__init__()
        # self.func = update_zeroPoint_someLayers
        self.func = lambda x:x
        self.block = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            ) 
        self.linear = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(4 * 4 * 256, 2048),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
            )

    def forward(self, x):
        x = self.block(x)
        x = x.view(-1, 4 * 4 * 256)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    m = alex_ori().eval()
    dummy = torch.randn(1, 3, 227, 227)
    torch.onnx.export(m, dummy, "alexnet_1.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input":{0:"N"}, "output":{0:"N"}})
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load("alexnet_1.onnx")),"alexnet.onnx")
    os.remove("alexnet_1.onnx")
