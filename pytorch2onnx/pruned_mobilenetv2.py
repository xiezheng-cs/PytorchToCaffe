import math

import numpy as np
import torch.nn as nn

__all__ = ['PrunedMobileNetV2', 'PrunedMobileBottleneck']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


def dwconv3x3(in_planes, out_planes, stride=1):
    "3x3 depth wise convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=in_planes, bias=False)


class PrunedMobileBottleneck(nn.Module):

    def __init__(self, in_planes, out_planes, expand=1, stride=1, pruning_rate=0, is_last=False):
        super(PrunedMobileBottleneck, self).__init__()
        self.name = "mobile-bottleneck"
        self.expand = expand

        intermedia_planes = in_planes * expand
        intermedia_planes = intermedia_planes - math.floor(intermedia_planes * pruning_rate)

        if is_last:
            out_planes = out_planes - math.floor(out_planes * pruning_rate)
        if self.expand != 1:
            self.conv1 = conv1x1(in_planes, intermedia_planes)
            self.bn1 = nn.BatchNorm2d(intermedia_planes)
            self.relu1 = nn.ReLU6(inplace=True)

        self.conv2 = dwconv3x3(
            intermedia_planes, intermedia_planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(intermedia_planes)
        self.relu2 = nn.ReLU6(inplace=True)

        self.conv3 = conv1x1(intermedia_planes, out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = (stride == 1 and in_planes == out_planes)
        self.block_index = 0

        if in_planes != out_planes and self.shortcut:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, out_planes),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.shortcut:
            residual = x

        out = x
        if self.expand != 1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut:
            if self.downsample is not None:
                out += self.downsample(residual)
            else:
                out += residual
            # out += residual

        return out


class PrunedMobileNetV2(nn.Module):
    """
    MobileNet_v2
    """

    def __init__(self, num_classes=1000, pruning_rate=0.0):
        super(PrunedMobileNetV2, self).__init__()

        block = PrunedMobileBottleneck
        # define network structure
        self.layer_width = np.array([32, 16, 24, 32, 64, 96, 160, 320])
        self.layer_width = np.around(self.layer_width * 1.0)
        self.layer_width = self.layer_width.astype(int)

        self.in_planes = self.layer_width[0].item()
        self.in_planes = self.in_planes - math.floor(float(self.in_planes) * pruning_rate)
        self.conv1 = conv3x3(3, self.in_planes, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu1 = nn.ReLU6(inplace=True)
        self.in_planes = self.layer_width[0].item()

        self.layer1 = self._make_layer(
            block, self.layer_width[1].item(), blocks=1, pruning_rate=pruning_rate)
        self.layer2 = self._make_layer(
            block, self.layer_width[2].item(), blocks=2, expand=6, stride=2, pruning_rate=pruning_rate)
        self.layer3 = self._make_layer(
            block, self.layer_width[3].item(), blocks=3, expand=6, stride=2, pruning_rate=pruning_rate)
        self.layer4 = self._make_layer(
            block, self.layer_width[4].item(), blocks=4, expand=6, stride=2, pruning_rate=pruning_rate)
        self.layer5 = self._make_layer(
            block, self.layer_width[5].item(), blocks=3, expand=6, stride=1, pruning_rate=pruning_rate)
        self.layer6 = self._make_layer(
            block, self.layer_width[6].item(), blocks=3, expand=6, stride=2, pruning_rate=pruning_rate)
        self.layer7 = self._make_layer(
            block, self.layer_width[7].item(), blocks=1, expand=6, pruning_rate=pruning_rate, is_last=True)

        self.in_planes = self.layer_width[7].item()
        self.in_planes = self.in_planes - math.floor(float(self.in_planes) * pruning_rate)
        fc_in_features = 1280
        fc_in_features = fc_in_features - math.floor(fc_in_features * pruning_rate)
        self.conv2 = conv1x1(in_planes=self.in_planes,
                             out_planes=fc_in_features)
        self.bn2 = nn.BatchNorm2d(fc_in_features)
        self.relu2 = nn.ReLU6(inplace=True)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0)

        self.fc = nn.Linear(fc_in_features, num_classes)
        # self.conv3 = conv1x1(1280, num_classes)
        # self.relu3 = nn.ReLU6(inplace=True)
        # self.fc = nn.Linear(self.layer_width[8], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_planes, blocks, expand=1, stride=1, pruning_rate=0.0, is_last=False):
        layers = []
        if is_last and blocks == 1:
            layers.append(block(self.in_planes, out_planes,
                                expand=expand, stride=stride, pruning_rate=pruning_rate, is_last=is_last))
        else:
            layers.append(block(self.in_planes, out_planes,
                                expand=expand, stride=stride, pruning_rate=pruning_rate))
        self.in_planes = out_planes
        for i in range(1, blocks):
            if is_last and i == blocks - 1:
                layers.append(block(self.in_planes, out_planes, expand=expand, pruning_rate=pruning_rate, is_last=is_last))
            else:
                layers.append(block(self.in_planes, out_planes, expand=expand, pruning_rate=pruning_rate))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward propagation
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # x = x.mean([2, 3])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)

        return x
