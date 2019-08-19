import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
from torchsummary import summary
import os

from dcp.utils.model_analyse import ModelAnalyse
from dcp.utils.logger import get_logger


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out

# LResNetxxE-IR or SE-LResNetxxE-IR
class ResNet(nn.Module):

    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.bn3 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)

        return x

# LResNet18E-IR
def resnet18(args, **kwargs):
    model = ResNet(IRBlock, [2, 2, 2, 2], use_se=args.use_se, **kwargs)
    return model

# LResNet34E-IR
def resnet34(args, **kwargs):
    model = ResNet(IRBlock, [3, 4, 6, 3], use_se=args.use_se, **kwargs)
    return model

# LResNet50E-IR
def resnet50(args, **kwargs):
    model = ResNet(IRBlock, [3, 4, 14, 3], use_se=args.use_se, **kwargs)
    return model

def resnet101(args, **kwargs):
    model = ResNet(IRBlock, [3, 4, 23, 3], use_se=args.use_se, **kwargs)
    return model

def resnet152(args, **kwargs):
    model = ResNet(IRBlock, [3, 8, 36, 3], use_se=args.use_se, **kwargs)
    return model

# SE-LResNet18E-IR
def resnet_face18(use_se=True, **kwargs):
    model = ResNet(IRBlock, [2, 2, 2, 2], use_se=use_se, **kwargs)
    return model

class MobileNet(nn.Module):
    def __init__(self, alpha):
        self.alpha = alpha
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, int(32 * self.alpha), 2),
            conv_dw(int(32 * self.alpha), int(64 * self.alpha), 1),
            conv_dw(int(64 * self.alpha), int(128 * self.alpha), 2),
            conv_dw(int(128 * self.alpha), int(128 * self.alpha), 1),
            conv_dw(int(128 * self.alpha), int(256 * self.alpha), 2),
            conv_dw(int(256 * self.alpha), int(256 * self.alpha), 1),
            conv_dw(int(256 * self.alpha), int(512 * self.alpha), 2),
            conv_dw(int(512 * self.alpha), int(512 * self.alpha), 1),
            conv_dw(int(512 * self.alpha), int(512 * self.alpha), 1),
            conv_dw(int(512 * self.alpha), int(512 * self.alpha), 1),
            conv_dw(int(512 * self.alpha), int(512 * self.alpha), 1),
            conv_dw(int(512 * self.alpha), int(512 * self.alpha), 1),
            conv_dw(int(512 * self.alpha), int(1024 * self.alpha), 2),
            conv_dw(int(1024 * self.alpha), int(1024 * self.alpha), 1),
        )
        self.bn2 = nn.BatchNorm2d(1024)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(1024 * 4 * 4, 512)
        self.bn3 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        return x


################################################  MobileFaceNet ########################################################
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(nn.Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=128, blocks = [1,4,6,2]):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))

        if blocks[0] == 1:
            self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        else:
            self.conv2_dw = Residual(64, num_block=blocks[0], groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        # GDC
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return out


# To do
# class GNAP_block(nn.Module):
#     def __init__(self, input_channel, embedding_size=128):
#         super(GNAP_block, self).__init__()
#
#
#         self.conv1 = nn.Conv2d(input_channel, embedding_size, kernel_size=(1,1), stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(embedding_size)
#         self.prelu = nn.PReLU(embedding_size)
#
#         self.bn2 = nn.BatchNorm2d(embedding_size)
#
#         self.pooling = nn.AvgPool2d(kernel_size=(7,7))
#         self.bn3 = nn.BatchNorm1d(embedding_size)
#
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.prelu(x)
#         return x


class ZQMobileFaceNet(nn.Module):
    def __init__(self, embedding_size=256, blocks=[4,8,16,4]):
        super(ZQMobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))

        # self.conv2_dw = Residual(64, num_block=blocks[0], groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        if blocks[0] == 1:
            self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        else:
            self.conv2_dw = Residual(64, num_block=blocks[0], groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(128, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(128, 256, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(256, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(256, 512, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(512, num_block=blocks[3], groups=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(512, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

        # GDC
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

        # GNAP


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return out


##################################  Arcface head #############################################################
class ArcMarginModel(nn.Module):
    r"""Implement of large margin arc distance: : cos(theta + m)
        """
    def __init__(self, args, num_classes, emb_size):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)

        if not self.training:
            return cosine

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # one_hot = torch.zeros(cosine.size()).cuda()
        one_hot = input.new_zeros(cosine.size())

        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

##################################  Cosface head #############################################################
class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: : cos(theta) - m
    https://arxiv.org/abs/1801.05599
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


if __name__ == "__main__":
    # model = MobileFaceNet(128, blocks=[1, 2, 2, 1])       # 2.36 M
    # model = MobileFaceNet(128, blocks=[1, 2, 3, 1])       # 2.62 M
    # model = MobileFaceNet(128, blocks=[1, 4, 6, 2])       # 3.83 M
    # model = MobileFaceNet(256, blocks=[2, 8, 16, 4])        # 7.61 M
    # model = MobileFaceNet(256, blocks=[4, 8, 16, 8])      # 8.75 M

    # zqcnn mobilefacenet_v2
    # model = ZQMobileFaceNet(256, blocks=[1, 4, 6, 2])         # 11.34 M
    # model = ZQMobileFaceNet(256, blocks=[4,8,16,4])         # 21.25 M
    # model = ZQMobileFaceNet(512, blocks=[4, 8, 16, 4])        # 21.75 M
    model = ZQMobileFaceNet(512, blocks=[4, 8, 16, 8])      # 29.88 M
    # model = ZQMobileFaceNet(512, blocks=[8, 16, 32, 8])     # 39.36 M

    print(model)
    summary(model, (3, 112, 112))

    save_path = './finetune-test'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = get_logger(save_path, "finetune-test")
    test_input = torch.randn(1, 3, 112, 112)
    model_analyse = ModelAnalyse(model, logger)
    model_analyse.flops_compute(test_input)
    # count = 0
    # for module in model.modules():
    #     print("{:d} = {}".format(count, module))
    #     count += 1
