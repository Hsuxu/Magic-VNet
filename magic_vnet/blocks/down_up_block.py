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


# class UpBlock(nn.Module):
#     """
#     Up sample block
#     """
#
#     def __init__(self, in_channels, out_channels,
#                  upper='interpolation',
#                  norm_type=nn.BatchNorm3d,
#                  act_type=nn.ReLU,
#                  se_type=None,
#                  drop_type=None,
#                  num_blocks=1,
#                  use_bottle_neck=False):
#         super(UpBlock, self).__init__()
#         assert upper in ['interpolation', 'upsample', 'convt'], "only 'interpolation'|'upsample'|'convt'  supported"
#         self.upper = upper
#         if self.upper == 'interpolation':
#             self.up_conv = ConvBnAct3d(in_channels, out_channels // 2,
#                                        norm_type=norm_type, act_type=act_type)
#         elif self.upper == 'upsample':
#             self.up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
#                                          ConvBnAct3d(in_channels, out_channels // 2,
#                                                      norm_type=norm_type, act_type=act_type))
#         elif self.upper == 'convt':
#             self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2)
#
#         self.drop = Drop(drop_type)
#         layers = []
#         block = ResBlock(out_channels, norm_type=norm_type, act_type=act_type, se_type=se_type)
#         if use_bottle_neck:
#             block = BottleNeck(out_channels, norm_type=norm_type, act_type=act_type, se_type=se_type)
#         for i in range(num_blocks):
#             layers.append(block)
#         self.res_block = nn.Sequential(*layers)
#
#     def forward(self, inputs, skip):
#         if self.upper == 'interpolation':
#             target_size = skip.size()[2:]
#             inputs = F.interpolate(inputs, size=target_size, mode='nearest')
#         input = self.up_conv(inputs)
#         out = torch.cat((input, skip), dim=1)
#         out = self.drop(out)
#         out = self.res_block(out)
#         return out


class UpBlock(nn.Module):
    """
    Upsample Block
    """

    def __init__(self, up_channels, ref_channels, out_channels,
                 upper='interpolation', norm_type=nn.BatchNorm3d,
                 act_type=nn.ReLU, se_type=None,
                 drop_type=None, num_blocks=1,
                 use_bottle_neck=False):
        super(UpBlock, self).__init__()

        assert upper in ['interpolation', 'upsample', 'convt'], "only 'interpolation'|'upsample'|'convt'  supported"
        self.upper = upper
        inner_channels = up_channels // 2
        if self.upper == 'interpolation':
            self.up_conv = ConvBnAct3d(up_channels, inner_channels,
                                       norm_type=norm_type, act_type=act_type)
        elif self.upper == 'upsample':
            self.up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                         ConvBnAct3d(up_channels, inner_channels,
                                                     norm_type=norm_type, act_type=act_type))
        elif self.upper == 'convt':
            self.up_conv = nn.Sequential(nn.ConvTranspose3d(up_channels, inner_channels,
                                                            kernel_size=2, stride=2, bias=False),
                                         ConvBnAct3d(inner_channels, inner_channels,
                                                     kernel_size=1, padding=0,
                                                     norm_type=norm_type,
                                                     act_type=act_type))

        self.trans_conv = ConvBnAct3d(inner_channels + ref_channels, out_channels, kernel_size=1,
                                      padding=0, norm_type=False, act_type=act_type)

        self.drop = Drop(drop_type)
        layers = []
        block = ResBlock(out_channels, norm_type=norm_type, act_type=act_type, se_type=se_type)
        if use_bottle_neck:
            block = BottleNeck(out_channels, norm_type=norm_type, act_type=act_type, se_type=se_type)
        for i in range(num_blocks):
            layers.append(block)
        self.res_block = nn.Sequential(*layers)

    def forward(self, up_tensor, ref_tensor):
        if isinstance(ref_tensor, torch.Tensor):
            ref_tensor = [ref_tensor]
        ref_tensor = torch.cat(ref_tensor, dim=1)
        if self.upper == 'interpolation':
            target_size = ref_tensor.size()[2:]
            up_tensor = F.interpolate(up_tensor, size=target_size, mode='nearest')
        up_tensor = self.up_conv(up_tensor)
        out = torch.cat([up_tensor, ref_tensor], dim=1)
        out = self.trans_conv(out)
        out = self.drop(out)
        out = self.res_block(out)
        return out
