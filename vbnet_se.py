import torch
import torch.nn as nn

from blocks import *


class VNet(nn.Module):
    """ v-net for segmentation """

    def __init__(self, in_channels, out_channels=2):
        super(VNet, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1, use_bottle_neck=False, use_se=True)
        self.down_64 = DownBlock(32, 2, use_bottle_neck=True, use_se=True)
        self.down_128 = DownBlock(64, 3, use_bottle_neck=True, use_se=True)
        self.down_256 = DownBlock(128, 3, use_bottle_neck=True, use_se=True)
        self.up_256 = UpBlock(256, 256, 3, use_bottle_neck=True, use_se=True)
        self.up_128 = UpBlock(256, 128, 3, use_bottle_neck=True, use_se=True)
        self.up_64 = UpBlock(128, 64, 2, use_bottle_neck=False, use_se=True)
        self.up_32 = UpBlock(64, 32, 1, use_bottle_neck=False, use_se=True)
        self.out_block = OutputBlock(32, out_channels)

    def forward(self, input):
        out16 = self.in_block(input)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)
        out = self.up_256(out256, out128)
        out = self.up_128(out, out64)
        out = self.up_64(out, out32)
        out = self.up_32(out, out16)
        out = self.out_block(out)
        return out

    def max_stride(self):
        return 16


def test():
    data = torch.rand(1, 1, 32, 320, 320)
    model = VNet(1, 2)
    out = model(data)
    print(out.size())

<<<<<<< HEAD
# test()
=======
# test()
>>>>>>> 5b9e7f37f44c9f0054dfedecde344f6ddbdd845c
