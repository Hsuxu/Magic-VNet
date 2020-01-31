import torch
import torch.nn as nn
import torch.nn.functional as F

from .convbnact import ConvBnAct3d


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type=nn.BatchNorm3d, act_type=nn.ReLU, ):
        super(InputBlock, self).__init__()
        self.conv = ConvBnAct3d(in_channels, out_channels,
                                kernel_size=5, padding=2, stride=1,
                                norm_type=norm_type,
                                act_type=act_type)

    def forward(self, input):
        out = self.conv(input)
        return out


class OutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type=nn.BatchNorm3d, act_type=nn.ReLU):
        super(OutBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvBnAct3d(in_channels, in_channels, norm_type=norm_type, act_type=act_type),
            ConvBnAct3d(in_channels, out_channels, kernel_size=1, padding=0, norm_type=False, act_type=False),
        )

    def forward(self, input):
        out = self.conv(input)
        return out
