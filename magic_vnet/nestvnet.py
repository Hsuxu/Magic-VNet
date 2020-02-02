import warnings
import torch
import torch.nn as nn

from .blocks import *

__all__ = ('NestVNet', 'NestVNet_CSE', 'NestVNet_SSE', 'NestVNet_SCSE', 'NestVNet_ASPP',
           'NestVBNet', 'NestVBNet_CSE', 'NestVBNet_SSE', 'NestVBNet_SCSE', 'NestVBNet_ASPP',
           'SK_NestVNet', 'SK_NestVNet_ASPP')


class NestVNet(nn.Module):
    def __init__(self, in_channels, num_class, **kwargs):
        super(NestVNet, self).__init__()
        norm_type = nn.BatchNorm3d
        act_type = nn.ReLU
        se_type = None
        drop_type = None
        feats = [16, 32, 64, 128, 256]
        num_blocks = [1, 2, 3, 3]
        block_name = 'residual'
        self._use_aspp = False
        self._deepsupervised = False
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
        if 'deep_supervised' in kwargs.keys():
            self._deepsupervised = kwargs['deep_supervised']

        self.in_conv = InputBlock(in_channels, feats[0],
                                  norm_type=norm_type,
                                  act_type=act_type)

        self.down1 = DownBlock(feats[0], feats[1], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[0], block_name=block_name)
        self.up1_1 = UpBlock(feats[1], feats[0], feats[0], norm_type=norm_type, act_type=act_type, se_type=se_type,
                             drop_type=drop_type, num_blocks=1, block_name=block_name)

        self.down2 = DownBlock(feats[1], feats[2], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[1], block_name=block_name)
        self.up2_1 = UpBlock(feats[2], feats[1], feats[1], norm_type=norm_type, act_type=act_type, se_type=se_type,
                             drop_type=drop_type, num_blocks=1, block_name=block_name)
        self.up2_2 = UpBlock(feats[1], 2 * feats[0], feats[0],
                             norm_type=norm_type, act_type=act_type, se_type=se_type,
                             drop_type=drop_type, num_blocks=1, block_name=block_name)

        self.down3 = DownBlock(feats[2], feats[3], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[2], block_name=block_name)
        self.up3_1 = UpBlock(feats[3], feats[2], feats[2], norm_type=norm_type, act_type=act_type, se_type=se_type,
                             drop_type=drop_type, num_blocks=1, block_name=block_name)
        self.up3_2 = UpBlock(feats[2], 2 * feats[1], feats[1], norm_type=norm_type, act_type=act_type, se_type=se_type,
                             drop_type=drop_type, num_blocks=1, block_name=block_name)
        self.up3_3 = UpBlock(feats[1], 3 * feats[0], feats[0], norm_type=norm_type, act_type=act_type, se_type=se_type,
                             drop_type=drop_type, num_blocks=1, block_name=block_name)

        self.down4 = DownBlock(feats[3], feats[4], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[3], block_name=block_name)
        self.up4_1 = UpBlock(feats[4], feats[3], feats[3], norm_type=norm_type, act_type=act_type, se_type=se_type,
                             drop_type=drop_type, num_blocks=1, block_name=block_name)
        self.up4_2 = UpBlock(feats[3], 2 * feats[2], feats[2], norm_type=norm_type, act_type=act_type, se_type=se_type,
                             drop_type=drop_type, num_blocks=1, block_name=block_name)
        self.up4_3 = UpBlock(feats[2], 3 * feats[1], feats[1], norm_type=norm_type, act_type=act_type, se_type=se_type,
                             drop_type=drop_type, num_blocks=1, block_name=block_name)
        self.up4_4 = UpBlock(feats[1], 4 * feats[0], feats[0], norm_type=norm_type, act_type=act_type, se_type=se_type,
                             drop_type=drop_type, num_blocks=1, block_name=block_name)

        if self._use_aspp:
            self.aspp = ASPP(feats[4], dilations=[1, 2, 3, 4], norm_type=norm_type, act_type=act_type,
                             drop_type=drop_type)

        self.out_block = [OutBlock(feats[0], num_class, norm_type, act_type)]
        if self._deepsupervised:
            self.out_block = [OutBlock(feats[0], num_class, norm_type, act_type)] * 4

    def forward(self, input):
        if input.size(2) // 16 == 0 or input.size(3) // 16 == 0 or input.size(4) // 16 == 0:
            raise RuntimeError("input tensor shape is too small")
        input = self.in_conv(input)
        down1 = self.down1(input)
        up1_1 = self.up1_1(down1, [input])

        down2 = self.down2(down1)
        up2_1 = self.up2_1(down2, [down1])
        up2_2 = self.up2_2(up2_1, [input, up1_1])

        down3 = self.down3(down2)
        up3_1 = self.up3_1(down3, down2)
        up3_2 = self.up3_2(up3_1, [down1, up2_1])
        up3_3 = self.up3_3(up3_2, [input, up1_1, up2_2])

        down4 = self.down4(down3)
        if self._use_aspp:
            down4 = self.aspp(down4)
        up4_1 = self.up4_1(down4, [down3])
        up4_2 = self.up4_2(up4_1, [down2, up3_1])
        up4_3 = self.up4_3(up4_2, [down1, up2_1, up3_2])
        up4_4 = self.up4_4(up4_3, [input, up1_1, up2_2, up3_3])

        out = self.out_block[0](up4_4)
        if self._deepsupervised:
            out3 = self.out_block[1](up3_3)
            out2 = self.out_block[2](up2_2)
            out1 = self.out_block[3](up1_1)
            out = torch.mean(torch.cat([out, out3, out2, out1], dim=1), dim=1, keepdim=True)
        return out


class NestVNet_CSE(NestVNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(NestVNet_CSE, self).__init__(in_channels, num_class, se_type='cse', **kwargs)


class NestVNet_SSE(NestVNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(NestVNet_SSE, self).__init__(in_channels, num_class, se_type='sse', **kwargs)


class NestVNet_SCSE(NestVNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(NestVNet_SCSE, self).__init__(in_channels, num_class, se_type='scse', **kwargs)


class NestVNet_ASPP(NestVNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(NestVNet_ASPP, self).__init__(in_channels, num_class, use_aspp=True, **kwargs)


class NestVBNet(NestVNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(NestVBNet, self).__init__(in_channels, num_class, block_name='bottleneck', **kwargs)


class NestVBNet_CSE(NestVBNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(NestVBNet_CSE, self).__init__(in_channels, num_class, block_name='bottleneck', se_type='cse', **kwargs)


class NestVBNet_SSE(NestVBNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(NestVBNet_SSE, self).__init__(in_channels, num_class, block_name='bottleneck', se_type='sse', **kwargs)


class NestVBNet_SCSE(NestVBNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(NestVBNet_SCSE, self).__init__(in_channels, num_class, block_name='bottleneck', se_type='scse', **kwargs)


class NestVBNet_ASPP(NestVBNet):
    def __init__(self, in_channels, num_class, **kwargs):
        super(NestVBNet_ASPP, self).__init__(in_channels, num_class, use_aspp=True, **kwargs)


class SK_NestVNet(NestVNet):
    def __init__(self, in_channels, num_class, **kwargs):
        if 'se_type' in kwargs.keys():
            warnings.warn('`se_type` keyword not working in `SK_NestVNet`', UserWarning)
        super(SK_NestVNet, self).__init__(in_channels, num_class, block_name='sk', **kwargs)


class SK_NestVNet_ASPP(NestVNet):
    def __init__(self, in_channels, num_class, **kwargs):
        if 'se_type' in kwargs.keys():
            warnings.warn('`se_type` keyword not working in `SK_NestVNet_ASPP`', UserWarning)
        super(SK_NestVNet_ASPP, self).__init__(in_channels, num_class, block_name='sk', use_aspp=True, **kwargs)
