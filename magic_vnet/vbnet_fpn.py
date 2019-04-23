import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from blocks import *

def vnet_kaiming_init(net):
    net.apply(kaiming_weight_init)

def vnet_focal_init(net, obj_p):
    net.apply(gaussian_weight_init)
    # initialize bias such as the initial predicted prob for objects are at obj_p.
    net.out_block.conv2.bias.data[1] = -np.log((1 - obj_p) / obj_p)

class VNet(nn.Module):
    """ v-net for segmentation """

    def __init__(self, in_channels, out_channels=2):
        super(VNet, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1, use_bottle_neck=False)
        self.down_64 = DownBlock(32, 2, use_bottle_neck=True)
        self.down_128 = DownBlock(64, 3, use_bottle_neck=True)
        self.down_256 = DownBlock(128, 3, use_bottle_neck=True)
        self.up_256 = UpBlock(256, 256, 3, use_bottle_neck=True)
        self.up_128 = UpBlock(256, 128, 3, use_bottle_neck=True)
        self.up_64 = UpBlock(128, 64, 2, use_bottle_neck=False)
        self.up_32 = UpBlock(64, 32, 1, use_bottle_neck=False)
        self.out_block = OutputBlock(32, out_channels)

        self.smooth128 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.smooth64 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.smooth32 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.post_block = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3,padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

    def forward(self, input):
        out16 = self.in_block(input)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)
        # print(out256.size(),out128.size())
        up128 = self.up_256(out256, out128)
        up64 = self.up_128(up128, out64)
        res64 = self._upsample_add(self.smooth128(up128), up64)
        up32 = self.up_64(up64, out32)
        res32 = self._upsample_add(self.smooth64(res64), up32)
        up32 = self.up_32(up32, out16)
        res32 = self._upsample_add(self.smooth32(res32), up32)
        res32 = self.post_block(res32)
        out = self.out_block(res32)
        return out

    def _upsample_add(self, x, y):
        D, H, W = y.size()[2:]
        resample_x = F.interpolate(x, size=(D, H, W), mode='trilinear', align_corners=True)
        return resample_x + y

    def max_stride(self):
        return 16

