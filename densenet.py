import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, with_se=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1)
        self.se = None
        if with_se:
            self.se = SELayer(growth_rate)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.se:
            x = self.se(x)
        return torch.cat((residual, x), dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n, with_se=False):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(n):
            layers.append(Bottleneck(channels, growth_rate, with_se=with_se))
            channels += growth_rate
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    stages = [6, 6, 6]

    def __init__(self, layers, growth_rate, in_channels=3, out_channels=10, reduction=0.5, with_se=False):
        super().__init__()
        channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(in_channels, channels,
                               kernel_size=3, stride=1, padding=1)

        self.block1 = DenseBlock(
            channels, growth_rate, layers[0], with_se=with_se)
        channels += layers[0] * growth_rate
        self.transition1 = Transition(channels, int(channels * reduction))
        channels = int(channels * reduction)

        self.block2 = DenseBlock(
            channels, growth_rate, layers[1], with_se=with_se)
        channels += layers[1] * growth_rate
        self.transition2 = Transition(channels, int(channels * reduction))
        channels = int(channels * reduction)

        self.block3 = DenseBlock(
            channels, growth_rate, layers[2], with_se=with_se)
        channels += layers[2] * growth_rate
        self.transition3 = Transition(channels, int(channels * reduction))
        channels = int(channels * reduction)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)

        x = self.block1(x)
        x = self.transition1(x)

        x = self.block2(x)
        x = self.transition2(x)

        x = self.block3(x)
        x = self.transition3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
