import torch
from torch import nn
from torch.nn import functional as F
from .mabn import MABN3d, CenConv3d

try:
    from inplace_abn import ABN, InPlaceABN, InPlaceABNSync
except:
    pass


class ConvBnAct3d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_type=ABN,
                 act_type=nn.ReLU):
        super(ConvBnAct3d, self).__init__()
        self.norm_type = norm_type
        self.act_type = act_type
        self.groups = groups
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              padding=padding,
                              stride=stride,
                              dilation=dilation,
                              groups=groups)
        # print(self.norm_type)
        if self.norm_type:
            if issubclass(self.norm_type, MABN3d):
                self.conv = CenConv3d(in_channels, out_channels, kernel_size,
                                      padding=padding,
                                      stride=stride,
                                      dilation=dilation,
                                      groups=groups)
            norm = self.__set_norm__(out_channels)
            self.norm = norm
        if self.act_type:
            self.act = act_type()

    def forward(self, input):
        out = self.conv(input)
        if self.norm_type:
            out = self.norm(out)
        if self.act_type:
            out = self.act(out)
        return out

    def __set_norm__(self, channels):

        norm = self.norm_type(channels)
        if issubclass(self.norm_type, ABN):
            if self.act_type:
                if issubclass(self.act_type, nn.ReLU):
                    act = 'relu'
                    self.act_type = False
                elif issubclass(self.act_type, nn.LeakyReLU):
                    act = 'leaky_relu'
                    self.act_type = False
                elif issubclass(self.act_type, nn.ELU):
                    act = 'elu'
                    self.act_type = False
                else:
                    act = 'identity'
                norm = self.norm_type(channels, activation=act)
            else:
                act = 'identity'
                norm = self.norm_type(channels, activation=act)
        if issubclass(self.norm_type, nn.GroupNorm):
            norm = self.norm_type(self.groups, channels)
        if issubclass(self.norm_type, MABN3d):
            norm = self.norm_type(channels)
        return norm


class BottConvBnAct3d(nn.Module):
    """Bottle neck structure"""

    def __init__(self, channels, ratio, norm_type=nn.BatchNorm3d, act_type=nn.ReLU, final_act=False):
        super(BottConvBnAct3d, self).__init__()
        self.conv1 = ConvBnAct3d(channels, channels // ratio, kernel_size=1, padding=0, norm_type=norm_type,
                                 act_type=act_type)
        self.conv2 = ConvBnAct3d(channels // ratio, channels // ratio, kernel_size=3, padding=1, norm_type=norm_type,
                                 act_type=act_type)
        if not final_act:
            self.conv3 = ConvBnAct3d(channels // ratio, channels, kernel_size=1, padding=0, norm_type=norm_type,
                                     act_type=False)
        else:
            self.conv3 = ConvBnAct3d(channels // ratio, channels, kernel_size=1, padding=0, norm_type=norm_type,
                                     act_type=act_type)

    def forward(self, input):
        out = self.conv3(self.conv2(self.conv1(input)))
        return out
