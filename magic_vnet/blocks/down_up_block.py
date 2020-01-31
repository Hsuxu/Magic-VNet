import torch
import torch.nn as nn
import torch.nn.functional as F

from .convbnact import ConvBnAct3d
from .drop_block import Drop
from .res_block import ResBlock, BottleNeck


class DownBlock(nn.Module):
    """
    Down-sample Block
    """

    def __init__(self, in_channels, out_channels,
                 norm_type=nn.BatchNorm3d,
                 act_type=nn.ReLU,
                 se_type=None,
                 drop_type=None,
                 num_blocks=1,
                 use_bottle_neck=False):
        super(DownBlock, self).__init__()
        self.down_conv = ConvBnAct3d(in_channels, out_channels, kernel_size=3, padding=1,
                                     stride=2, norm_type=norm_type, act_type=act_type)
        layers = []
        block = ResBlock(out_channels, norm_type=norm_type, act_type=act_type, se_type=se_type)
        if use_bottle_neck:
            block = BottleNeck(out_channels, norm_type=norm_type, act_type=act_type, se_type=se_type)
        for i in range(num_blocks):
            layers.append(block)
        self.res_block = nn.Sequential(*layers)

        self.drop = Drop(drop_type)

    def forward(self, input):
        out = self.res_block(self.down_conv(input))
        out = self.drop(out)
        return out


class UpBlock(nn.Module):
    """
    Up sample block
    """

    def __init__(self, in_channels, out_channels,
                 upper='interpolation',
                 norm_type=nn.BatchNorm3d,
                 act_type=nn.ReLU,
                 se_type=None,
                 drop_type=None,
                 num_blocks=1,
                 use_bottle_neck=False):
        super(UpBlock, self).__init__()
        assert upper in ['interpolation', 'upsample', 'convt'], "only 'interpolation'|'upsample'|'convt'  supported"
        self.upper = upper
        if self.upper == 'interpolation':
            self.up_conv = ConvBnAct3d(in_channels, out_channels // 2,
                                       norm_type=norm_type, act_type=act_type)
        elif self.upper == 'upsample':
            self.up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                         ConvBnAct3d(in_channels, out_channels // 2,
                                                     norm_type=norm_type, act_type=act_type))
        elif self.upper == 'convt':
            self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2)

        self.drop = Drop(drop_type)
        layers = []
        block = ResBlock(out_channels, norm_type=norm_type, act_type=act_type, se_type=se_type)
        if use_bottle_neck:
            block = BottleNeck(out_channels, norm_type=norm_type, act_type=act_type, se_type=se_type)
        for i in range(num_blocks):
            layers.append(block)
        self.res_block = nn.Sequential(*layers)

    def forward(self, input, skip):
        if self.upper == 'interpolation':
            target_size = skip.size()[2:]
            input = F.interpolate(input, size=target_size, mode='nearest')
        input = self.up_conv(input)
        out = torch.cat((input, skip), dim=1)
        out = self.drop(out)
        out = self.res_block(out)
        return out
