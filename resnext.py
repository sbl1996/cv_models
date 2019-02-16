import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1, groups=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias)


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        channels = in_channels // reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size()[:2]
        s = self.squeeze(x).view(b, c)
        s = self.excitation(s).view(b, c, 1, 1)
        return x * s


class Bottleneck(nn.Module):
    def __init__(self, in_channels, channels, out_channels, stride=1, groups=32, with_se=True):
        super().__init__()
        if groups == -1:
            groups = channels
        self.with_se = with_se
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels, stride=stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if with_se:
            self.se = SELayer(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.with_se:
            x = self.se(x)

        x = x + self.shortcut(residual)
        x = self.relu3(x)
        return x


class ResNeXt(nn.Module):
    cfg = [32, 64, 128, 256, 512]

    def __init__(self, layers, in_channels=3, num_classes=10, groups=32, with_se=True, dropout=0):
        super().__init__()
        self.conv = conv3x3(in_channels, self.cfg[0], bias=True)

        self.layer1 = make_layer(
            Bottleneck, layers[0], self.cfg[0], self.cfg[1], self.cfg[2], stride=1, groups=groups, with_se=with_se)
        self.layer2 = make_layer(
            Bottleneck, layers[1], self.cfg[2], self.cfg[2], self.cfg[3], stride=2, groups=groups, with_se=with_se)
        self.layer3 = make_layer(
            Bottleneck, layers[2], self.cfg[3], self.cfg[3], self.cfg[4], stride=2, groups=groups, with_se=with_se)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = None
        if dropout != 0:
            self.dropout = nn.Dropout(dropout, inplace=True)
        self.fc = nn.Linear(self.cfg[4], num_classes)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        if self.dropout:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def make_layer(block, num_layers, in_channels, channels, out_channels, stride, groups, with_se):
    layers = []
    layers.append(block(in_channels, channels, out_channels,
                        stride=stride, groups=groups, with_se=with_se))
    for i in range(1, num_layers):
        layers.append(
            block(out_channels, channels, out_channels, groups=groups, with_se=with_se))
    return nn.Sequential(*layers)
