import torch
import torch.nn as nn

try:
    from .blocks import *
    from .utils import init_parameters
except:
    from blocks import *
    from utils import init_parameters


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type=nn.BatchNorm3d, act_type=nn.ReLU, ):
        super(InputBlock, self).__init__()
        self.conv = ConvBnAct3d(in_channels, out_channels,
                                kernel_size=5, padding=2, stride=1,
                                norm_type=norm_type,
                                act_type=act_type)

    def forward(self, input):
        out = self.conv(input)
        return out


class DownBlock(BasicDownBlock):
    """
    Down-sample Block
    """

    def __init__(self, in_channels, out_channels, norm_type=nn.BatchNorm3d,
                 act_type=nn.ReLU, se_type=None, drop_type=None, num_blocks=1):
        super(DownBlock, self).__init__()
        self.down_conv = ConvBnAct3d(in_channels, out_channels, kernel_size=3, padding=1,
                                     stride=2, norm_type=norm_type, act_type=act_type)
        layers = []
        for i in range(num_blocks):
            layers.append(ResBlock(out_channels, norm_type=norm_type, act_type=act_type, se_type=se_type))
        self.res_block = nn.Sequential(*layers)
        self.drop = Drop(drop_type)


class UpBlock(BasicUpBlock):
    def __init__(self, in_channels, out_channels, upper='interpolation', norm_type=nn.BatchNorm3d,
                 act_type=nn.ReLU, se_type=None, drop_type=None, num_blocks=1):
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
        for i in range(num_blocks):
            layers.append(ResBlock(out_channels, norm_type=norm_type, act_type=act_type, se_type=se_type))
        self.res_block = nn.Sequential(*layers)


class VNet(nn.Module):
    def __init__(self, in_channels, num_class, norm_type=nn.BatchNorm3d, act_type=nn.ReLU, **kwargs):
        super(VNet, self).__init__()
        feats = [16, 32, 64, 128, 256]
        se_type = None
        num_blocks = [1, 2, 3, 3]
        if 'feats' in kwargs.keys():
            feats = kwargs['feats']
        if 'se_type' in kwargs.keys():
            se_type = kwargs['se_type']
        if 'num_blocks' in kwargs.keys():
            num_blocks = kwargs['num_blocks']
        self.in_conv = InputBlock(in_channels, feats[0],
                                  norm_type=norm_type,
                                  act_type=act_type)

        self.down1 = DownBlock(feats[0], feats[1], norm_type=norm_type,
                               act_type=act_type, se_type=se_type, num_blocks=num_blocks[0])
        self.down2 = DownBlock(feats[1], feats[2], norm_type=norm_type,
                               act_type=act_type, se_type=se_type, num_blocks=num_blocks[1])
        self.down3 = DownBlock(feats[2], feats[3], norm_type=norm_type,
                               act_type=act_type, se_type=se_type, num_blocks=num_blocks[2])
        self.down4 = DownBlock(feats[3], feats[4], norm_type=norm_type,
                               act_type=act_type, se_type=se_type, num_blocks=num_blocks[3])

        self.up4 = UpBlock(feats[4], feats[4], norm_type=norm_type,
                           act_type=act_type, se_type=se_type, num_blocks=num_blocks[3])

        self.up3 = UpBlock(feats[4], feats[3], norm_type=norm_type,
                           act_type=act_type, se_type=se_type, num_blocks=num_blocks[2])

        self.up2 = UpBlock(feats[3], feats[2], norm_type=norm_type,
                           act_type=act_type, se_type=se_type, num_blocks=num_blocks[1])

        self.up1 = UpBlock(feats[2], feats[1], norm_type=norm_type,
                           act_type=act_type, se_type=se_type, num_blocks=num_blocks[0])

        if num_class == 2:
            num_class = 1

        self.out_block = OutBlock(feats[1], num_class, norm_type, act_type)

        init_parameters(self)

    def forward(self, input):
        if input.size(2) // 16 == 0 or input.size(3) // 16 == 0 or input.size(4) // 16 == 0:
            raise RuntimeError("input tensor shape is too small")
        input = self.in_conv(input)
        down1 = self.down1(input)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        up4 = self.up4(down4, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, input)

        out = self.out_block(up1)
        return out


class VNet_CSE(VNet):
    def __init__(self, in_channels, num_class, norm_type=nn.BatchNorm3d, act_type=nn.ReLU, **kwargs):
        super(VNet_CSE, self).__init__(in_channels, num_class, norm_type, act_type,
                                       se_type='cse')


class VNet_SSE(VNet):
    def __init__(self, in_channels, num_class, norm_type=nn.BatchNorm3d, act_type=nn.ReLU, **kwargs):
        super(VNet_SSE, self).__init__(in_channels, num_class, norm_type, act_type,
                                       se_type='sse')


class VNet_SCSE(VNet):
    def __init__(self, in_channels, num_class, norm_type=nn.BatchNorm3d, act_type=nn.ReLU, **kwargs):
        super(VNet_SCSE, self).__init__(in_channels, num_class, norm_type, act_type,
                                        se_type='scse')


if __name__ == '__main__':
    data = torch.rand((1, 1, 32, 32, 32))
    model = VNet_CSE(1, 2)
    out = model(data)
    print(out.shape)
