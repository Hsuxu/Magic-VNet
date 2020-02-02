import torch
import torch.nn as nn

from ..convbnact import ConvBnAct3d


class SKConv3d(nn.Module):
    """"
    Basic select kernel operation
    Reference:
    Args:
        channels(int): input and output channel dimensionality.
        branch(int): the number of branches Default 2
        ratio(int): the radio for compute d, the length of z, Default 4
        min_channels(int): the minimum dim of the vector z in paper, Default 32paper)
        stride(int): stride of convolution, Default 1.
        groups(int): num of convolution groups, Default 1.
        norm_type(type): normalization method, Default `nn.BatchNorm3d`
        act_type(type): activation function, Default `nn.ReLU`

    Shapes:
        input: input tensor `(N, C_{in}, D_{in}, H_{in}, W_{in})`
        output: output tensor `(N, C_{out}, D_{out}, H_{out}, W_{out})`, where `D_{out}=D_{in}`
    """

    def __init__(self, channels,
                 branch=2,
                 ratio=4,
                 stride=1,
                 groups=1,
                 min_channels=32,
                 norm_type=nn.BatchNorm3d,
                 act_type=nn.ReLU):
        super(SKConv3d, self).__init__()
        dim = max(channels // ratio, min_channels)
        self.branch = branch
        self.convs = nn.ModuleList([])
        for i in range(branch):
            self.convs.append(
                ConvBnAct3d(channels, channels, kernel_size=3 + i * 2,
                            padding=i + 1, stride=stride, groups=groups,
                            norm_type=norm_type,
                            act_type=act_type)
            )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(channels, dim)
        self.fcs = nn.ModuleList([])
        for i in range(branch):
            self.fcs.append(nn.Linear(dim, channels))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        for i, conv in enumerate(self.convs):
            fea = conv(input).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_sum = torch.sum(feas, dim=1)
        fea_avg = self.avg_pool(fea_sum)

        fea_avg = fea_avg.view(fea_avg.size(0), -1)
        fea_fc = self.fc(fea_avg)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_fc).unsqueeze_(dim=1)
            if i == 0:
                attn_vec = vector
            else:
                attn_vec = torch.cat([attn_vec, vector], dim=1)
        attn_vec = self.softmax(attn_vec)
        n = feas.dim() - attn_vec.dim()
        for _ in range(n):
            attn_vec = attn_vec.unsqueeze(-1)
        out = (feas * attn_vec).sum(dim=1)
        return out


class SK_Block(nn.Module):
    """ select kernel block
    Args:
        in_channels: input channel dimensionality.
        out_channels: output channel dimensionality.
        branch: the number of branches Default 2
        groups: num of convolution groups, Default 1.
        stride: stride of convolution, Default 1.
        ratio: the radio for compute d, the length of z, Default 4
        min_channels: the minimum dim of the vector z in paper, Default 32paper)
        norm_type: normalization method, Default `nn.BatchNorm3d`
        act_type: activation function, Default `nn.ReLU`
    Shapes:
        input: input tensor `(N, C_{in}, D_{in}, H_{in}, W_{in})`
        output: output tensor `(N, C_{out}, D_{out}, H_{out}, W_{out})`
    """

    def __init__(self, in_channels, out_channels,
                 branch=2, groups=1, ratio=4, stride=1, min_channels=32,
                 norm_type=nn.BatchNorm3d, act_type=nn.ReLU):
        super(SK_Block, self).__init__()
        inner_channels = out_channels // 2
        self.trans1 = ConvBnAct3d(in_channels, inner_channels,
                                  kernel_size=1, padding=0,
                                  norm_type=norm_type,
                                  act_type=False)
        self.skconv = nn.Sequential(SKConv3d(inner_channels, branch=branch,
                                             ratio=ratio, min_channels=min_channels,
                                             groups=groups, stride=stride,
                                             norm_type=norm_type, act_type=act_type),
                                    norm_type(inner_channels))
        self.trans2 = ConvBnAct3d(inner_channels, out_channels,
                                  kernel_size=1, padding=0,
                                  norm_type=norm_type,
                                  act_type=False)
        self.short_cut = nn.Sequential()
        if in_channels != out_channels:
            self.short_cut = ConvBnAct3d(in_channels, out_channels,
                                         kernel_size=1, padding=0,
                                         stride=stride, groups=1,
                                         norm_type=norm_type, act_type=False)

        self.final_act = nn.Sequential()
        if act_type:
            self.final_act = act_type()

    def forward(self, input):
        out = self.trans1(input)
        out = self.skconv(out)
        out = self.trans2(out)
        out = out + self.short_cut(input)
        out = self.final_act(out)
        return out
