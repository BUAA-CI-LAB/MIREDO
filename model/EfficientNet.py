# this file is prepared for project 026
# Created by iboxl

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.k1 = nn.Parameter(torch.ones(int(planes // 2)), requires_grad=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes, bias=False)
        self.k2 = nn.Parameter(torch.ones(int(planes // 2)), requires_grad=True)                      
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.k3 = nn.Parameter(torch.ones(int(out_planes // 2)), requires_grad=True)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.k_short = nn.Parameter(torch.ones(int(out_planes // 2)), requires_grad=True)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        self.fc1 = nn.Conv2d(out_planes, out_planes//4, kernel_size=1)
        self.kf1 = nn.Parameter(torch.ones(int(out_planes//4 // 2)), requires_grad=True)
        self.fc2 = nn.Conv2d(out_planes//4, out_planes, kernel_size=1)
        self.kf2 = nn.Parameter(torch.ones(int(out_planes // 2)), requires_grad=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut(x) if self.stride == 1 else out
        # Squeeze-Excitation
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = self.fc2(w).sigmoid()
        out = out * w + shortcut
        return out

    def forward_balanceInfer(self, func_update, inputs):
        # out = F.relu(self.bn1(self.conv1(x)))
        w = func_update(self.conv1.weight, self.k1)
        out = F.conv2d(inputs, w, stride=self.conv1.stride, padding = self.conv1.padding)
        out = F.relu(self.bn1(out))
        # out = F.relu(self.bn2(self.conv2(out)))
        w = func_update(self.conv2.weight, self.k2)
        out = F.conv2d(out, w, stride=self.conv2.stride, padding = self.conv2.padding, groups=self.conv2.groups)
        out = F.relu(self.bn2(out))
        # out = self.bn3(self.conv3(out))
        w = func_update(self.conv3.weight, self.k3)
        out = F.conv2d(out, w, stride=self.conv3.stride, padding = self.conv3.padding)
        out = self.bn3(out)
        # shortcut = self.shortcut(inputs) if self.stride == 1 else out
        if self.stride == 1:
            if len(self.shortcut) > 1:
                w = self.shortcut[0].weight
                w = func_update(w, self.k_short)
                shortcut = F.conv2d(inputs, w, stride=self.shortcut[0].stride, padding = self.shortcut[0].padding)
                shortcut = self.shortcut[1](shortcut)
            else :
                shortcut = inputs
        else :
            shortcut = out
        # Squeeze-Excitation
        # multi = F.avg_pool2d(out, out.size(2))
        multi = F.adaptive_avg_pool2d(out, 1)
        # w = F.relu(self.fc1(w))
        w = func_update(self.fc1.weight, self.kf1)
        multi = F.conv2d(multi, w, bias=self.fc1.bias, stride=self.fc1.stride, padding = self.fc1.padding)
        multi = F.relu(multi)
        # w = self.fc2(w).sigmoid()
        w = func_update(self.fc2.weight, self.kf2)
        multi = F.conv2d(multi, w, bias=self.fc2.bias, stride=self.fc2.stride, padding = self.fc2.padding)
        multi = multi.sigmoid()
        out = out * multi + shortcut
        return out

class Block_redu(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block_redu, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        self.fc1 = nn.Conv2d(out_planes, out_planes//4, kernel_size=1)
        self.fc2 = nn.Conv2d(out_planes//4, out_planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut(x) if self.stride == 1 else out
        # Squeeze-Excitation
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = self.fc2(w).sigmoid()
        out = out * w + shortcut
        return out

class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, pretrained:bool = False, **kwargs):
        super(EfficientNet, self).__init__()
        self.cfg = [
            (1,  16, 1, 2),
            (6,  24, 2, 1),
            (6,  40, 2, 2),
            (6,  80, 3, 2),
            (6, 112, 3, 1),
            (6, 192, 4, 2),
            (6, 320, 1, 2)]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(self.cfg[-1][1], num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
