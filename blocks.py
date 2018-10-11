import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu3(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding, do_act=True):
        super(ConvBnRelu3, self).__init__()
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
    def __init__(self, channels, ratio, do_act=True):
        super(BottConvBnRelu3, self).__init__()
        self.conv1 = ConvBnRelu3(channels, channels // ratio, ksize=1, padding=0, do_act=True)
        self.conv2 = ConvBnRelu3(channels // ratio, channels // ratio, ksize=3, padding=1, do_act=True)
        self.conv3 = ConvBnRelu3(channels // ratio, channels, ksize=1, padding=0, do_act=do_act)

    def forward(self, input):
        out = self.conv3(self.conv2(self.conv1(input)))
        return out


class ResidualBlock3(nn.Module):
    """ residual block with variable number of convolutions """

    def __init__(self, channels, ksize, padding, num_convs):
        super(ResidualBlock3, self).__init__()

        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(ConvBnRelu3(channels, channels, ksize, padding, do_act=True))
            else:
                layers.append(ConvBnRelu3(channels, channels, ksize, padding, do_act=False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):

        output = self.ops(input)
        return self.act(input + output)


class BottResidualBlock3(nn.Module):
    """ block with bottle neck conv"""

    def __init__(self, channels, ratio, num_convs):
        super(BottResidualBlock3, self).__init__()
        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(BottConvBnRelu3(channels, ratio, True))
            else:
                layers.append(BottConvBnRelu3(channels, ratio, False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.ops(input)
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

    def __init__(self, in_channels, num_convs, use_bottle_neck=False):
        super(DownBlock, self).__init__()
        out_channels = in_channels * 2
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, groups=1)
        self.down_bn = nn.BatchNorm3d(out_channels)
        self.down_act = nn.ReLU(inplace=True)
        if use_bottle_neck:
            self.rblock = BottResidualBlock3(out_channels, 4, num_convs)
        else:
            self.rblock = ResidualBlock3(out_channels, 3, 1, num_convs)

    def forward(self, input):
        out = self.down_act(self.down_bn(self.down_conv(input)))
        out = self.rblock(out)
        return out


class UpBlock(nn.Module):
    """ Upsample block of v-net """

    def __init__(self, in_channels, out_channels, num_convs, use_bottle_neck=False):
        super(UpBlock, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2, groups=1)
        self.up_bn = nn.BatchNorm3d(out_channels // 2)
        self.up_act = nn.ReLU(inplace=True)
        if use_bottle_neck:
            self.rblock = BottResidualBlock3(out_channels, 4, num_convs)
        else:
            self.rblock = ResidualBlock3(out_channels, 3, 1, num_convs)

    def forward(self, input, skip):
        out = self.up_act(self.up_bn(self.up_conv(input)))
        out = torch.cat((out, skip), 1)
        out = self.rblock(out)
        return out


class AttentionGate(nn.Module):
    """
    Attention Gate 
    """
    def __init__(self, in_channels, gate_channels, inner_channels, mode='softmax'):
        super(AttentionGate, self).__init__()
        self.mode = mode
        self.conv_gate = nn.Conv3d(in_channels=gate_channels, out_channels=inner_channels, kernel_size=1, stride=1,
                                   padding=0)
        self.conv_fea = nn.Conv3d(in_channels=in_channels, out_channels=inner_channels, kernel_size=1, stride=1,
                                  padding=0)
        self.conv_mask = nn.Conv3d(in_channels=inner_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv_W = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, feature, gate):
        fea_x = self.conv_fea(feature)
        gating_g = self.conv_gate(gate)
        fea_x_size = fea_x.size()[2:]
        gating_g = F.interpolate(gating_g, size=fea_x_size, mode='trilinear', align_corners=True)
        F_fea_gate = F.leaky_relu(gating_g + fea_x, negative_slope=0.01)
        mask = self.conv_mask(F_fea_gate)
        # default softmax
        mask_tmp = mask.view(mask.size(0), 1, -1)
        mask_tmp = F.softmax(mask_tmp, dim=2)
        mask_tmp = mask_tmp.view_as(mask)
        if self.mode == 'sigmoid':
            mask_tmp = F.sigmoid(mask)
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
        
