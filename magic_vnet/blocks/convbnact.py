import torch
from torch import nn
from torch.nn import functional as F
from .mabn import MABN3d, CenConv3d


class ConvBnAct3d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_type=nn.BatchNorm3d,
                 act_type=nn.ReLU):
        super(ConvBnAct3d, self).__init__()
        self.norm_type = norm_type
        self.act_type = act_type
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        if norm_type:
            if issubclass(self.norm_type, MABN3d):
                self.conv = CenConv3d(in_channels, out_channels, kernel_size,
                                      padding=padding,
                                      stride=stride,
                                      dilation=dilation,
                                      groups=groups)
            self.norm = norm_type(out_channels)
        if self.act_type:
            self.act = act_type()

    def forward(self, input):
        out = self.conv(input)
        if self.norm_type:
            out = self.norm(out)
        if self.act_type:
            out = self.act(out)
        return out


class BottConvBnAct3d(nn.Module):
    """Bottle neck structure"""

    def __init__(self, channels, ratio, norm_type=nn.BatchNorm3d, act_type=nn.ReLU, final_act=False):
        super(BottConvBnAct3d, self).__init__()
        self.conv1 = ConvBnAct3d(channels, channels // ratio, kernel_size=1, padding=0, norm_type=norm_type,
                                 act_type=act_type)
        self.conv2 = ConvBnAct3d(channels // ratio, channels // ratio, kernel_size=3, padding=1, norm_type=norm_type,
                                 act_type=act_type)
        if not final_act:
            self.conv3 = ConvBnAct3d(channels // ratio, channels, kernel_size=1, padding=0, norm_type=norm_type,
                                     act_type=False)
        else:
            self.conv3 = ConvBnAct3d(channels // ratio, channels, kernel_size=1, padding=0, norm_type=norm_type,
                                     act_type=act_type)

    def forward(self, input):
        out = self.conv3(self.conv2(self.conv1(input)))
        return out
