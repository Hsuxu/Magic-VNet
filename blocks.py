import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu3(nn.Module):
    """ classic combination: conv + batch normalization [+ relu] """

    def __init__(self, in_channels, out_channels, ksize=3, padding=1, do_act=True):
        super(ConvBnRelu3, self).__init__()
        if isinstance(ksize, tuple) and padding is None:
            padding = tuple(item // 2 for item in ksize)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=ksize, padding=padding, groups=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.do_act = do_act
        if do_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.bn(self.conv(input))
        if self.do_act:
            out = self.act(out)
        return out


class BottConvBnRelu3(nn.Module):
    """Bottle neck structure"""

    def __init__(self, channels, ratio, do_act=True):
        super(BottConvBnRelu3, self).__init__()
        self.conv1 = ConvBnRelu3(channels, channels // ratio, ksize=1, padding=0, do_act=True)
        self.conv2 = ConvBnRelu3(channels // ratio, channels // ratio, ksize=3, padding=1, do_act=True)
        self.conv3 = ConvBnRelu3(channels // ratio, channels, ksize=1, padding=0, do_act=do_act)

    def forward(self, input):
        out = self.conv3(self.conv2(self.conv1(input)))
        return out


class SeBlock(nn.Module):
    """
    reference:
    Squeeze-and-Excitation Networks
    Dual Attention Network for Scene Segmentation
    channel wise attention still
    """

    def __init__(self, channels, ratio=8):
        super(SeBlock, self).__init__()
        # if not channels//ratio >0:
        #     raise ValueError
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // ratio),
                                nn.LeakyReLU(),
                                nn.Linear(channels // ratio, channels),
                                nn.Sigmoid())

    def forward(self, input):
        bs, chn, dim, hei, wid = input.size()
        scale = self.avg_pool(input)
        scale = scale.view(bs, -1)
        scale = self.fc(scale)
        scale = scale.view(bs, chn, 1, 1, 1)
        return scale * input


class ResidualBlock3(nn.Module):
    """ residual block with variable number of convolutions """

    def __init__(self, channels, ksize, padding, num_convs, use_se=False):
        super(ResidualBlock3, self).__init__()
        self.use_se = use_se
        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(ConvBnRelu3(channels, channels, ksize, padding, do_act=True))
            else:
                layers.append(ConvBnRelu3(channels, channels, ksize, padding, do_act=False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)
        if self.use_se:
            self.se_block = SeBlock(channels, ratio=8)
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        output = self.ops(input)
        if self.use_se:
            output = self.se_block(output)
            output = self.gamma * output
        return self.act(input + output)


class BottResidualBlock3(nn.Module):
    """ block with bottle neck conv"""

    def __init__(self, channels, ratio, num_convs, use_se=False):
        super(BottResidualBlock3, self).__init__()
        self.use_se = use_se
        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(BottConvBnRelu3(channels, ratio, True))
            else:
                layers.append(BottConvBnRelu3(channels, ratio, False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)
        if self.use_se:
            self.se_block = SeBlock(channels, ratio=8)
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        output = self.ops(input)
        if self.use_se:
            output = self.se_block(output)
            output = self.gamma * output
        return self.act(input + output)


class InputBlock(nn.Module):
    """ input block of vb-net """

    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out


class DownBlock(nn.Module):
    """ downsample block of v-net """

    def __init__(self, in_channels, num_convs, use_bottle_neck=False, use_se=False):
        super(DownBlock, self).__init__()
        out_channels = in_channels * 2
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, groups=1)
        self.down_bn = nn.BatchNorm3d(out_channels)
        self.down_act = nn.ReLU(inplace=True)
        if use_bottle_neck:
            self.rblock = BottResidualBlock3(out_channels, 4, num_convs, use_se=use_se)
        else:
            self.rblock = ResidualBlock3(out_channels, 3, 1, num_convs, use_se=use_se)

    def forward(self, input):
        out = self.down_act(self.down_bn(self.down_conv(input)))
        out = self.rblock(out)
        return out


class UpBlock(nn.Module):
    """ Upsample block of v-net """

    def __init__(self, in_channels, out_channels, num_convs, use_bottle_neck=False, use_se=False):
        super(UpBlock, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2, groups=1)
        self.up_bn = nn.BatchNorm3d(out_channels // 2)
        self.up_act = nn.ReLU(inplace=True)
        if use_bottle_neck:
            self.rblock = BottResidualBlock3(out_channels, 4, num_convs, use_se=use_se)
        else:
            self.rblock = ResidualBlock3(out_channels, 3, 1, num_convs, use_se=use_se)

    def forward(self, input, skip):
        out = self.up_act(self.up_bn(self.up_conv(input)))
        out = torch.cat((out, skip), 1)
        out = self.rblock(out)
        return out


class OutputBlock(nn.Module):
    """ output block of v-net

        The output is a list of foreground-background probability vectors.
        The length of the list equals to the number of voxels in the volume
    """

    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        out = self.act1(self.bn1(self.conv1(input)))
        out = self.conv2(out)
        out = self.softmax(out)
        return out


class AttentionGate(nn.Module):
    """
    Attention Gate
    """

    def __init__(self, in_channels, gate_channels, inner_channels, mode='softmax'):
        super(AttentionGate, self).__init__()
        self.mode = mode
        self.conv_gate = nn.Conv3d(in_channels=gate_channels,
                                   out_channels=inner_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.conv_fea = nn.Conv3d(in_channels=in_channels,
                                  out_channels=inner_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.bn_relu = nn.Sequential(nn.BatchNorm3d(inner_channels),
                                     nn.ReLU())
        self.conv_mask = nn.Conv3d(in_channels=inner_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv_W = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU())
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature, gate):
        fea_x = self.conv_fea(feature)
        gating_g = self.conv_gate(gate)

        fea_x_size = fea_x.size()[2:]
        gating_g = F.interpolate(gating_g, size=fea_x_size, mode='trilinear', align_corners=True)
        F_fea_gate = self.bn_relu(gating_g + fea_x)
        mask = self.conv_mask(F_fea_gate)
        # default softmax
        mask_tmp = mask.view(mask.size(0), -1)
        mask_tmp = F.softmax(mask_tmp, dim=1)
        mask_tmp = mask_tmp.view_as(mask)
        if self.mode == 'sigmoid':
            mask_tmp = self.sigmoid(mask)
        mask = mask_tmp
        mask = mask.expand_as(feature)
        fea_xl = torch.matmul(mask, feature)
        return self.conv_W(fea_xl)


def kaiming_weight_init(m, bn_std=0.02):
    classname = m.__class__.__name__
    if 'Conv3d' in classname or 'ConvTranspose3d' in classname:
        if int(torch.__version__.split('.')[1]) < 4:
            nn.init.kaiming_normal(m.weight)
        else:
            nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, bn_std)
        m.bias.data.zero_()


def gaussian_weight_init(m, conv_std=0.01, bn_std=0.01):
    classname = m.__class__.__name__
    if 'Conv3d' in classname or 'ConvTranspose3d' in classname:
        m.weight.data.normal_(0, conv_std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, bn_std)
        m.bias.data.zero_()

# class MyTest(object):
#     def __init__(self):
#         self.data = torch.rand(1, 1, 32, 64, 64)
#
#     def test_InputBlock(self):
#         model = InputBlock(self.data.size(1), out_channels=self.data.size(1))
#         out = model(self.data)
#         print(out.size() == self.data.size())
#
#     def test_downblock(self):
#         model = DownBlock(1, 3)
#         out = model(self.data)
#         shape = self.data.size()
#         shape = [s // 2 for s in shape[2:]]
#         print(list(out.size())[2:] == shape)


# if __name__ == '__main__':
#     # test = MyTest()
#     # test.test_downblock()
#     gate = torch.rand(1, 16, 16, 32, 32)
#     fea = torch.rand(1, 8, 32, 64, 64)
#     model = AttentionGate(8, 16, 4)
#     out = model(fea, gate)
#     print(out.size(),fea.size())
#
