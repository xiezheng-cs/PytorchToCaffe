#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/27 16:00
# @Author  : xiezheng
# @Site    : 
# @File    : insightface_mobilefacenet.py


import math
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from torch.nn import Parameter


class pruned_Bottleneck_mobilefacenet(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expansion, pruning_rate=0.0):
        super(pruned_Bottleneck_mobilefacenet, self).__init__()

        self.connect = stride == 1 and in_planes == out_planes
        planes = in_planes * expansion

        self.pruned_channel_plane = int(planes - math.floor(planes * pruning_rate))

        # original
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.prelu1 = nn.PReLU(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.prelu2 = nn.PReLU(planes)
        # self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_planes)

        # pruned
        self.conv1 = nn.Conv2d(in_planes, self.pruned_channel_plane, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.pruned_channel_plane)
        self.prelu1 = nn.PReLU(self.pruned_channel_plane)
        self.conv2 = nn.Conv2d(self.pruned_channel_plane, self.pruned_channel_plane, kernel_size=3, stride=stride,
                               padding=1, groups=self.pruned_channel_plane, bias=False)
        self.bn2 = nn.BatchNorm2d(self.pruned_channel_plane)
        self.prelu2 = nn.PReLU(self.pruned_channel_plane)
        self.conv3 = nn.Conv2d(self.pruned_channel_plane, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.connect:
            return x + out
        else:
            return out


class pruned_Mobilefacenet(nn.Module):

    Mobilefacenet_bottleneck_setting = [
        # [t, c , n ,s] = [expansion, out_planes, num_blocks, stride]
        [2, 64, 5, 2],
        [4, 128, 1, 2],
        [2, 128, 6, 1],
        [4, 128, 1, 2],
        [2, 128, 2, 1]
    ]

    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting, pruning_rate=0.0):
        super(pruned_Mobilefacenet, self).__init__()
        self.inplanes = 64

        # self.pruned_channel_plane = int(512 - math.floor(512 * pruning_rate))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, groups=64, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.prelu2 = nn.PReLU(64)

        self.layers = self._make_layer(pruned_Bottleneck_mobilefacenet, bottleneck_setting, pruning_rate)

        # self.conv3 = nn.Conv2d(128, int(512 - math.floor(512 * pruning_rate)), kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(int(512 - math.floor(512 * pruning_rate)))
        # self.prelu3 = nn.PReLU(int(512 - math.floor(512 * pruning_rate)))
        # self.conv4 = nn.Conv2d(int(512 - math.floor(512 * pruning_rate)), int(512 - math.floor(512 * pruning_rate)),
        #                        kernel_size=7, groups=int(512 - math.floor(512 * pruning_rate)), stride=1, padding=0, bias=False)
        # self.bn4 = nn.BatchNorm2d(int(512 - math.floor(512 * pruning_rate)))
        # self.linear = nn.Linear(int(512 - math.floor(512 * pruning_rate)), 128, bias=False)
        # self.bn5 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.prelu3 = nn.PReLU(512)
        self.conv4 = nn.Conv2d(512, 512,kernel_size=7, groups=512, stride=1, padding=0,bias=False)
        self.bn4 = nn.BatchNorm2d(512 )
        self.linear = nn.Linear(512, 128, bias=False)
        self.bn5 = nn.BatchNorm1d(128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, setting, pruning_rate=0.0):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t, pruning_rate))
                else:
                    layers.append(block(self.inplanes, c, 1, t, pruning_rate))
                self.inplanes = c
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = self.layers(out)
        out = self.prelu3(self.bn3(self.conv3(out)))
        out = self.bn4(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = self.bn5(self.linear(out))
        return out


if __name__ == "__main__":
    # model = pruned_Mobilefacenet()                       # Params size (MB):3.83 M, Total params: 1,003,136
    model = pruned_Mobilefacenet(pruning_rate=0.25)    # Total params: 753,888
    # model = pruned_Mobilefacenet(pruning_rate=0.3)
    # model = pruned_Mobilefacenet(pruning_rate=0.5)       #  Total params: 504,640

    # print(model.state_dict())
    # print("---------------------")
    # for key in model.state_dict().keys():
    #     print(key)
    print(model)
    summary(model, (3, 112, 112))
