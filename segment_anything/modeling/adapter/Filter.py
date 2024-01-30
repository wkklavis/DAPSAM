
import torch
import torch.nn as nn

from typing import Type
import torch
import torch.fft as fft
import pywt



class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        output = self.sigmoid(max_result + avg_result)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class FilterFFT(nn.Module):
    def __init__(self, kernel_size=7, channel=768, reduction=16):
        super().__init__()
        self.channel = channel*2
        self.con_proj = nn.Conv2d(self.channel, self.channel, kernel_size=1)
        self.bn = nn.SyncBatchNorm(self.channel)
        self.relu = nn.ReLU()
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        N, C, H, W = x.shape
        # res = x
        x = fft.fftn(x)
        x = torch.concat([x.real, x.imag], 1)
        x = self.con_proj(x)
        x = self.bn(x)
        x = self.relu(x)
        # SpatialAttention
        output = self.sa(x)
        x = x*output
        x = torch.split(x, int(self.channel/2), 1)
        x = torch.complex(x[0], x[1])
        x = fft.ifftn(x, dim=(-2, -1)).real
        return x

class SpatialFilter(nn.Module):
    def __init__(self, kernel_size=7, channel=3072,reduction=16):
        super().__init__()
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        # SpatialAttention
        output = self.sa(x)
        x = x*output
        return x

class ChannelFilter(nn.Module):
    def __init__(self, channel=768):
        super().__init__()
        self.ca = ChannelAttention(channel)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # SpatialAttention
        output = self.ca(x)
        x = x*output
        x = x.permute(0, 2, 3, 1)
        return x

