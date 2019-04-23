import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from blocks import *


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rate):
        super(ASPP, self).__init__()
        if rate == 1:
            ksize = 1
            padding = 0
        else:
            ksize = 3
            padding = rate
        self.atrous_conv = nn.Conv3d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=ksize,
                                     stride=1,
                                     padding=padding,
                                     dilation=rate,
                                     bias=False)
        self.atrous_bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.atrous_conv(input)
        out = self.relu(self.atrous_bn(out))
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


def vnet_kaiming_init(net):
    net.apply(kaiming_weight_init)


def vnet_focal_init(net, obj_p):
    net.apply(gaussian_weight_init)
    # initialize bias such as the initial predicted prob for objects are at obj_p.
    net.out_block.conv2.bias.data[1] = -np.log((1 - obj_p) / obj_p)


class VNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNet, self).__init__()
        rates = [1, 3, 5, 7]
        self.input = InputBlock(in_channels, 16)
        self.down32 = DownBlock(16, 1, use_bottle_neck=False)
        self.down64 = DownBlock(32, 2, use_bottle_neck=False)
        self.down128 = DownBlock(64, 3, use_bottle_neck=True)
        self.down256 = DownBlock(128, 3, use_bottle_neck=True)
        self.aspp1 = ASPP(256, 64, rate=rates[0])
        self.aspp2 = ASPP(256, 64, rate=rates[1])
        self.aspp3 = ASPP(256, 64, rate=rates[2])
        self.aspp4 = ASPP(256, 64, rate=rates[3])
        self.avg_pool1 = nn.Sequential(nn.AdaptiveAvgPool3d(1))
        self.pool_conv = nn.Sequential(nn.Conv3d(256, 64, 1, stride=1),
                                       nn.ReLU())
        self.trans = nn.Sequential(nn.Conv3d(64 * 5, 256, 1),
                                   nn.BatchNorm3d(256),
                                   nn.ReLU())
        # self.up256 = UpBlock(256, 256, 5, use_bottle_neck=True)
        self.up256 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.ConvTranspose3d(256, 256, 2, stride=2),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            ConvBnRelu3(256, 256, 3, 1)
        )
        self.up128 = UpBlock(256, 128, 3, use_bottle_neck=True)
        # self.up64 = UpBlock(128, 64, 3, use_bottle_neck=False)
        self.up64 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.ConvTranspose3d(128, 64, 2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            ConvBnRelu3(64, 64, 3, 1)
        )
        self.up32 = UpBlock(64, 32, 1, use_bottle_neck=False)

        # self.transition_input = ConvBnRelu3(16, 16, 1, padding=0)

        self.outblock = OutputBlock(32, out_channels)

    def forward(self, input):
        input = self.input(input)
        down32 = self.down32(input)
        down64 = self.down64(down32)
        down128 = self.down128(down64)
        down256 = self.down256(down128)
        pool1 = self.avg_pool1(down256)
        pool1 = self.pool_conv(pool1)
        x1 = self.aspp1(down256)
        x2 = self.aspp2(down256)
        x3 = self.aspp3(down256)
        x4 = self.aspp4(down256)
        x5 = F.interpolate(pool1, size=down256.size()[2:], mode='trilinear', align_corners=True)
        down320 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        down256 = self.trans(down320)
        out = self.up256(down256)
        out = self.up128(out, down64)
        out = self.up64(out)
        out = self.up32(out, input)
        out = self.outblock(out)
        return out

    def max_stride(self):
        return 16


# def test():
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#     net = VNet(1, 2).cuda()
#     for i in range(30):
#         data = torch.rand(1, 1, 32, 320, 320).cuda()
#         # aspp = ASPP(1, 1, 2).cuda()
#         # out = aspp(data)
#         # print(out.size())
#         out = net(data)
#         print(out.size())
#
#
# test()
