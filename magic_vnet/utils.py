import torch.nn as nn


def init_parameters(net):
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight.data)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
