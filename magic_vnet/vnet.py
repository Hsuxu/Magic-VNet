import warnings
import torch
import torch.nn as nn

from .blocks import *

__all__ = ('VNet', 'VNet_CSE', 'VNet_SSE', 'VNet_SCSE', 'VNet_ASPP',
           'VBNet', 'VBNet_CSE', 'VBNet_SSE', 'VBNet_SCSE', 'VBNet_ASPP',
           'SKVNet', 'SKVNet_ASPP')


class VNet(nn.Module):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VNet, self).__init__()
        norm_type = nn.BatchNorm3d
        act_type = nn.ReLU
        se_type = None
        drop_type = None
        feats = [16, 32, 64, 128, 256]
        num_blocks = [1, 2, 3, 3]
        block_name = 'residual'
        self._use_aspp = False
        if num_class == 2:
            num_class = 1
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
        if 'block_name' in kwargs.keys():
            block_name = kwargs['block_name']

        self.in_conv = InputBlock(in_channels, feats[0],
                                  norm_type=norm_type,
                                  act_type=act_type)

        self.down1 = DownBlock(feats[0], feats[1], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[0], block_name=block_name)
        self.down2 = DownBlock(feats[1], feats[2], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[1], block_name=block_name)
        self.down3 = DownBlock(feats[2], feats[3], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[2], block_name=block_name)
        self.down4 = DownBlock(feats[3], feats[4], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[3], block_name=block_name)
        if self._use_aspp:
            self.aspp = ASPP(feats[4], dilations=[1, 2, 3, 4], norm_type=norm_type, act_type=act_type,
                             drop_type=drop_type)
        self.up4 = UpBlock(feats[4], feats[3], feats[4], norm_type=norm_type, act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[3], block_name=block_name)
        self.up3 = UpBlock(feats[4], feats[2], feats[3], norm_type=norm_type, act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[2], block_name=block_name)
        self.up2 = UpBlock(feats[3], feats[1], feats[2], norm_type=norm_type, act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[1], block_name=block_name)
        self.up1 = UpBlock(feats[2], feats[0], feats[1], norm_type=norm_type, act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[0], block_name=block_name)

        self.out_block = OutBlock(feats[1], num_class, norm_type, act_type)

        init_weights(self)

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


class VNet_CSE(VNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VNet_CSE, self).__init__(in_channels, num_class, se_type='cse', **kwargs)


class VNet_SSE(VNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VNet_SSE, self).__init__(in_channels, num_class, se_type='sse', **kwargs)


class VNet_SCSE(VNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VNet_SCSE, self).__init__(in_channels, num_class, se_type='scse', **kwargs)


class VNet_ASPP(VNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VNet_ASPP, self).__init__(in_channels, num_class, use_aspp=True, **kwargs)


class VBNet(VNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VBNet, self).__init__(in_channels, num_class, block_name='bottleneck', **kwargs)


class VBNet_CSE(VBNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VBNet_CSE, self).__init__(in_channels, num_class, block_name='bottleneck', se_type='cse', **kwargs)


class VBNet_SSE(VBNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VBNet_SSE, self).__init__(in_channels, num_class, block_name='bottleneck', se_type='sse', **kwargs)


class VBNet_SCSE(VBNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VBNet_SCSE, self).__init__(in_channels, num_class, block_name='bottleneck', se_type='scse', **kwargs)


class VBNet_ASPP(VBNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(VBNet_ASPP, self).__init__(in_channels, num_class, block_name='bottleneck', use_aspp=True, **kwargs)


class SKVNet(VNet):
    def __init__(self, in_channels, num_class, **kwargs):
        if 'se_type' in kwargs.keys():
            warnings.warn('`se_type` keyword not working in `SKVNet`', UserWarning)
        super(SKVNet, self).__init__(in_channels, num_class, block_name='sk', **kwargs)


class SKVNet_ASPP(VNet):
    def __init__(self, in_channels, num_class, **kwargs):
        if 'se_type' in kwargs.keys():
            warnings.warn('`se_type` keyword not working in `SKVNet_ASPP`', UserWarning)
        super(SKVNet_ASPP, self).__init__(in_channels, num_class, block_name='sk', use_aspp=True, **kwargs)
