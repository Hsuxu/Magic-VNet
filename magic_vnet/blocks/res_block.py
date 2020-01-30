import torch
from torch import nn

try:
    from .convbnact import ConvBnAct3d, BottConvBnAct3d
    from .squeeze_excitation import ChannelSELayer3D, SpatialSELayer3D, SpatialChannelSELayer3D
except:
    from convbnact import ConvBnAct3d, BottConvBnAct3d
    from squeeze_excitation import ChannelSELayer3D, SpatialSELayer3D, SpatialChannelSELayer3D


class BasicResBlock(nn.Module):
    """
    Base residual block
    """

    def forward(self, input):
        out = self.residual(input)
        if self.final_act:
            out = self.final_act(input + out)
        else:
            out = input + out
        return out


class BottleNeck(BasicResBlock):
    """Bottle neck structure"""

    def __init__(self, channels, ratio=4, norm_type=nn.BatchNorm3d, act_type=nn.ReLU, se_type=None, num_layer=2):
        super(BottleNeck, self).__init__()
        layers = []
        for i in range(num_layer):
            if i == num_layer - 1:
                conv = BottConvBnAct3d(channels, ratio=ratio, norm_type=norm_type, act_type=act_type, final_act=True)
            else:
                conv = BottConvBnAct3d(channels, ratio=ratio, norm_type=norm_type, act_type=act_type, final_act=False)
            layers.append(conv)
        if se_type == 'cse':
            cse_module = ChannelSELayer3D(channels, act_type=act_type)
            layers.append(cse_module)
        elif se_type == 'sse':
            sse_module = SpatialSELayer3D(channels)
            layers.append(sse_module)
        elif se_type == 'scse':
            scse_module = SpatialChannelSELayer3D(channels, act_type=act_type)
            layers.append(scse_module)

        self.residual = nn.Sequential(*layers)
        self.final_act = act_type()


class ResBlock(BasicResBlock):
    """
    ResBlock with 'cse'|'sse'|'scse'
    """

    def __init__(self, channels, norm_type=nn.BatchNorm3d, act_type=nn.ReLU, se_type=None, num_layer=1):
        super(ResBlock, self).__init__()
        assert se_type in ['cse', 'sse', 'scse', None], "se_type should be in ['cse', 'sse', 'scse', None]"
        layers = []

        conv = ConvBnAct3d(channels, channels, norm_type=norm_type, act_type=act_type)

        for i in range(num_layer):
            if i == num_layer - 1:
                conv = ConvBnAct3d(channels, channels, norm_type=norm_type, act_type=False)
                layers.append(conv)
            else:
                layers.append(conv)
            if se_type == 'cse':
                cse_module = ChannelSELayer3D(channels, act_type=act_type)
                layers.append(cse_module)
            elif se_type == 'sse':
                sse_module = SpatialSELayer3D(channels)
                layers.append(sse_module)
            elif se_type == 'scse':
                scse_module = SpatialChannelSELayer3D(channels)
                layers.append(scse_module)
        self.residual = nn.Sequential(*layers)
        self.final_act = act_type()


def test():
    model = ResBlock(32, se_type='cse', norm_type=nn.BatchNorm3d)
    data = torch.rand((1, 32, 64, 64, 64))
    out = model(data)
    print(out.size())


if __name__ == '__main__':
    test()
