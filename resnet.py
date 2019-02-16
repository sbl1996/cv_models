import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            )

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x + self.shortcut(residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        channels = self.out_channels // self.expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1)

        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels, stride=stride)

        self.bn3 = nn.BatchNorm2d(channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(
                in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        if self.downsample:
            x = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out += x
        return out


class ResNet(nn.Module):
    stages = [16, 16, 32, 64]

    def __init__(self, block, layers, k=4, num_classes=10, **kwargs):
        super().__init__()
        self.block_kwargs = kwargs
        self.conv = nn.Conv2d(
            3, self.stages[0], kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(
            block, self.stages[0] * 1, self.stages[1] * k, layers[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(
            block, self.stages[1] * k, self.stages[2] * k, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(
            block, self.stages[2] * k, self.stages[3] * k, layers[2], stride=2, **kwargs)

        self.bn = nn.BatchNorm2d(self.stages[3] * k)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stages[3] * k, num_classes)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1, **kwargs):
        layers = []
        layers.append(block(in_channels, out_channels,
                            stride=stride, **kwargs))
        for i in range(1, blocks):
            layers.append(
                block(out_channels, out_channels, **kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x.mean()
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        channels = in_channels // reduction
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size()[:2]
        s = self.avgpool(x).view(b, c)
        s = self.layers(s).view(b, c, 1, 1)
        return x * s


class SEBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.se = SELayer(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            )

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.se(x)
        return x + self.shortcut(residual)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        channels = self.out_channels // self.expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1)

        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels, stride=stride)

        self.bn3 = nn.BatchNorm2d(channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1)
        self.se = SELayer(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(
                in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        if self.downsample:
            x = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        out = self.se(out)

        out += x
        return out


class WideSEBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, with_se=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

        self.se = None
        if with_se:
            self.se = SELayer(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(
                in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        residual = x
        o1 = self.relu1(self.bn1(x))
        z = self.conv1(o1)
        o2 = self.relu2(self.bn2(z))
        z = self.conv2(o2)
        if self.se:
            z = self.se(z)
        if self.downsample:
            residual = self.downsample(o1)
        return z + residual
