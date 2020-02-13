import torch
import torch.nn as nn
import torch.nn.functional as F


class CenConv2d(nn.Module):
    """Conv2d layer with Weight Centralization.
    The args is exactly same as torch.nn.Conv2d. It's suggested to set bias=False when
    using CenConv2d with MABN.
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(CenConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_planes))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class CenConv3d(nn.Module):
    """Conv2d layer with Weight Centralization.
    The args is exactly same as torch.nn.Conv2d. It's suggested to set bias=False when
    using CenConv2d with MABN.
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(CenConv3d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_planes))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
