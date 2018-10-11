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
    """ v-net with attention for segmentation """

    def __init__(self, in_channels, num_classes=2):
        super(VNet, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1, use_bottle_neck=False)
        self.down_64 = DownBlock(32, 2, use_bottle_neck=True)
        self.down_128 = DownBlock(64, 3, use_bottle_neck=True)
        self.down_256 = DownBlock(128, 3, use_bottle_neck=True)
        self.up_256 = UpBlock(256, 256, 3, use_bottle_neck=True)
        self.up_128 = UpBlock(256, 128, 3, use_bottle_neck=True)
        self.up_64 = UpBlock(128, 64, 2, use_bottle_neck=True)
        self.up_32 = UpBlock(64, 32, 1, use_bottle_neck=True)
        self.out_block = OutputBlock(32, num_classes)

        self.att256 = AttentionGate(in_channels=128, gate_channels=256, inner_channels=64)
        self.att128 = AttentionGate(in_channels=64, gate_channels=256, inner_channels=32)
        self.att64 = AttentionGate(in_channels=32, gate_channels=128, inner_channels=16)
        self.att32 = AttentionGate(in_channels=16, gate_channels=64, inner_channels=8)

    def forward(self, input):
        out16 = self.in_block(input)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)

        att128 = self.att256(out128, out256)
        up128 = self.up_256(out256, att128)

        att64 = self.att128(out64, up128)
        up64 = self.up_128(up128, att64)

        att32 = self.att64(out32, up64)
        up32 = self.up_64(up64, att32)

        att16 = self.att32(out16, up32)
        up16 = self.up_32(up32, att16)

        out = self.out_block(up16)
        return out


# def test():
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     data = torch.rand(1, 1, 32, 320, 320)
#     net = VNet(in_channels=1, num_classes=2)
#     net(data)
#
#
# if __name__ == '__main__':
#     test()
