import torch
import torch.nn as nn

from .blocks import *


class VBNet(nn.Module):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VBNet, self).__init__()
        norm_type = nn.BatchNorm3d
        act_type = nn.ReLU
        se_type = None
        drop_type = None
        feats = [16, 32, 64, 128, 256]
        num_blocks = [1, 2, 3, 3]
        self._use_aspp = False
        if 'norm_type' in kwargs.keys():
            norm_type = kwargs['norm_type']
        if 'act_type' in kwargs.keys():
            act_type = kwargs['act_type']
        if 'feats' in kwargs.keys():
            feats = kwargs['feats']
        if 'se_type' in kwargs.keys():
            se_type = kwargs['se_type']
        if 'num_blocks' in kwargs.keys():
            num_blocks = kwargs['num_blocks']
        if 'drop_type' in kwargs.keys():
            drop_type = kwargs['drop_type']
        if 'use_aspp' in kwargs.keys():
            self._use_aspp = kwargs['use_aspp']

        self.in_conv = InputBlock(in_channels, feats[0],
                                  norm_type=norm_type,
                                  act_type=act_type)

        self.down1 = DownBlock(feats[0], feats[1], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[0], use_bottle_neck=True)
        self.down2 = DownBlock(feats[1], feats[2], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[1], use_bottle_neck=True)
        self.down3 = DownBlock(feats[2], feats[3], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[2], use_bottle_neck=True)
        self.down4 = DownBlock(feats[3], feats[4], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[3], use_bottle_neck=True)

        self.up4 = UpBlock(feats[4], feats[4], norm_type=norm_type, act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[3], use_bottle_neck=True)
        self.up3 = UpBlock(feats[4], feats[3], norm_type=norm_type, act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[2], use_bottle_neck=True)
        self.up2 = UpBlock(feats[3], feats[2], norm_type=norm_type, act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[1], use_bottle_neck=True)
        self.up1 = UpBlock(feats[2], feats[1], norm_type=norm_type, act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[0], use_bottle_neck=True)

        if num_class == 2:
            num_class = 1

        self.out_block = OutBlock(feats[1], num_class, norm_type, act_type)

    def forward(self, input):
        if input.size(2) // 16 == 0 or input.size(3) // 16 == 0 or input.size(4) // 16 == 0:
            raise RuntimeError("input tensor shape is too small")
        input = self.in_conv(input)
        down1 = self.down1(input)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        if self._use_aspp:
            down4 = self.aspp(down4)

        up4 = self.up4(down4, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, input)

        out = self.out_block(up1)
        return out


class VBNet_CSE(VBNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VBNet_CSE, self).__init__(in_channels, num_class, se_type='cse', **kwargs)


class VBNet_SSE(VBNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VBNet_SSE, self).__init__(in_channels, num_class, se_type='sse', **kwargs)


class VBNet_SCSE(VBNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VBNet_SCSE, self).__init__(in_channels, num_class, se_type='scse', **kwargs)


class VBNet_ASPP(VBNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VBNet_ASPP, self).__init__(in_channels, num_class, use_aspp=True, **kwargs)


if __name__ == '__main__':
    data = torch.rand((1, 1, 32, 32, 32))
    model = VBNet_CSE(1, 2)
    out = model(data)
    print(out.shape)
