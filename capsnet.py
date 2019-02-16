import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(s):
    r"""
    Inputs:
        s: (*, n)
    Outputs:
        v: (*, n)
    """
    norm = torch.norm(s, dim=-1, keepdim=True)
    norm_square = norm ** 2
    v = (norm / (1 + norm_square)) * s
    return v


def routing(u_ji, r):
    r"""
    Inputs:
        u_j|i: (batch_size, n_in, n_out, d_out)
        r: number of iterations
    """
    batch_size, n_in, n_out, d_out = u_ji.size()
    b = torch.zeros(batch_size, n_in, n_out)
    for i in range(r):
        c = torch.softmax(b, dim=-1)
        s = torch.sum(c.unsqueeze(-1) * u_ji, dim=1)
        v = squash(s)  # (batch_size, n_out, d_out)
        b += (u_ji.view(batch_size, n_in, n_out, 1, d_out) @
              v.view(batch_size, 1, n_out, d_out, 1)).squeeze()
    return v


class Capsule(nn.Module):
    r"""
    Args:
        n_in: number of input capsule
        d_in: dimension of input capsule
        n_out: number of output capsule
        d_out: dimension of output capsule
        r: number of iterations for routing algorithm
    Inputs:
        u_i: (batch_size, n_in, d_in)
    Outputs:
        v_j: (batch_size, n_out, d_out)
    """

    def __init__(self, n_in, d_in, n_out, d_out, r):
        super().__init__()
        self.r = r
        self.weight = nn.Parameter(torch.randn(1, n_in, n_out, d_in, d_out))
        self.bias = nn.Parameter(torch.randn(n_in, n_out, 1))

    def forward(self, u_i):
        batch_size, n_in, d_in = u_i.size()
        u_ji = u_i.view(batch_size, n_in, 1, 1, d_in) @ self.weight
        u_ji = u_ji.squeeze()
        u_ji += self.bias
        v = routing(u_ji, self.r)
        return v


class CapsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=9, stride=2)
        self.caps = Capsule(6*6*32, 8, 10, 16, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size(0)
        x = x.view(b, 6*6*32, 8)
        x = self.caps(x)
        return x


def capsule_margin_loss(v, t, margin_positive=0.9, margin_negative=0.1, down_weighting=0.5):
    r"""
    Inputs:
        v: (batch_size, C, d)
        t: (batch_size)
    """
    a = torch.norm(v, dim=-1)
    indices = torch.arange(v.size(0))
    loss_positive = torch.mean(torch.relu(
        margin_positive - a[indices, t]) ** 2)
    mask = torch.ones_like(a)
    mask[indices, t] = 0
    loss_negative = torch.mean((torch.relu(a - margin_negative) ** 2) * mask)
    loss = loss_positive + down_weighting * loss_negative
    return loss
