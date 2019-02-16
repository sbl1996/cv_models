import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1, groups=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride=1):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        channels = in_channels * expansion

        if expansion == 1:
            self.conv = nn.Sequential(
                conv3x3(in_channels, channels, stride=stride, groups=channels),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
                conv1x1(channels, out_channels),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                conv1x1(in_channels, channels),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
                conv3x3(channels, channels, stride=stride, groups=channels),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
                conv1x1(channels, out_channels),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    cfg = [
        # t, c, n, s
        (1,  16, 1, 1),
        (6,  24, 2, 2),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = [conv3x3(in_channels, 32)]
        in_channels = 32
        for t, c, n, s in self.cfg:
            self.features += make_layer(Bottleneck, n, in_channels, c, s, t)
            in_channels = c
        self.features = nn.Sequential(*self.features)
        self.conv = conv1x1(in_channels, 1280)
        self.bn = nn.BatchNorm2d(1280)
        self.relu = nn.ReLU6(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def make_layer(block, num_layers, in_channels, out_channels, stride, expansion):
    layers = []
    layers.append(block(in_channels, out_channels,
                        stride=stride, expansion=expansion))
    for i in range(1, num_layers):
        layers.append(
            block(out_channels, out_channels, expansion=expansion))
    return layers
