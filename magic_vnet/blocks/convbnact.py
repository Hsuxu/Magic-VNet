import torch
from torch import nn
from torch.nn import functional as F


class ConvBnAct3d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 dilation=1,
                 norm_type=nn.BatchNorm3d,
                 act_type=nn.ReLU):
        super(ConvBnAct3d, self).__init__()
        self.norm_type = norm_type
        self.act_type = act_type
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation)
        if norm_type:
            # if isinstance(norm_type, nn.BatchNorm3d):
            #     if act_type:
            #         if isinstance(act_type, nn.ReLU):
            #             act = 'relu'
            #             self.act_type = False
            #         elif isinstance(act_type, nn.LeakyReLU):
            #             act = 'leaky_relu'
            #             self.act_type = False
            #         elif isinstance(act_type, nn.ELU):
            #             act = 'elu'
            #             self.act_type = False
            #         else:
            #             act = 'none'
            #     else:
            #         act = 'none'
            #     self.norm = norm_type(out_channels, activation=act)
            # else:
            #     self.norm = norm_type(out_channels)
            self.norm = norm_type(out_channels)
        if self.act_type:
            self.act = act_type()
        self._init_weights()

    def forward(self, input):
        out = self.conv(input)
        if self.norm_type:
            out = self.norm(out)
        if self.act_type:
            out = self.act(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)


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
