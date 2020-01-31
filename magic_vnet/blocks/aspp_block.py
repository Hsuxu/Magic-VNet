import torch
import torch.nn as nn
import torch.nn.functional as F

from .convbnact import ConvBnAct3d
from .drop_block import Drop


class ASPP(nn.Module):
    def __init__(self, channels, ratio=4,
                 dilations=[1, 2, 3, 4],
                 norm_type=nn.BatchNorm3d,
                 act_type=nn.ReLU,
                 drop_type=None):
        super(ASPP, self).__init__()
        assert dilations[0] == 1, 'The first item in dilations should be `1`'
        inner_channels = channels // ratio
        cat_channels = inner_channels * 5
        self.aspp0 = ConvBnAct3d(channels, inner_channels, kernel_size=1, padding=0,
                                 dilation=dilations[0], norm_type=norm_type, act_type=act_type)
        self.aspp1 = ConvBnAct3d(channels, inner_channels, kernel_size=3, padding=dilations[1],
                                 dilation=dilations[1], norm_type=norm_type, act_type=act_type)
        self.aspp2 = ConvBnAct3d(channels, inner_channels, kernel_size=3, padding=dilations[2],
                                 dilation=dilations[2], norm_type=norm_type, act_type=act_type)
        self.aspp3 = ConvBnAct3d(channels, inner_channels, kernel_size=3, padding=dilations[3],
                                 dilation=dilations[3], norm_type=norm_type, act_type=act_type)
        self.avg_conv = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                      ConvBnAct3d(channels, inner_channels, kernel_size=1,
                                                  padding=0, norm_type=False, act_type=act_type))
        self.transition = ConvBnAct3d(cat_channels, channels, kernel_size=1, padding=0,
                                      norm_type=norm_type, act_type=act_type)

        self.drop = Drop(drop_type)

    def forward(self, input):
        aspp0 = self.aspp0(input)
        aspp1 = self.aspp1(input)
        aspp2 = self.aspp2(input)
        aspp3 = self.aspp3(input)
        avg = self.avg_conv(input)
        avg = F.interpolate(avg, aspp3.size()[2:], mode='nearest')
        out = torch.cat((aspp0, aspp1, aspp2, aspp3, avg), dim=1)
        out = self.transition(out)
        return out
