import torch
import torch.nn as nn
import torch.nn.functional as F

from .dropblock.dropblock import DropBlock3D

__all__ = ('Drop', 'keep_origin')


def keep_origin(input, **kwargs):
    return input


class Drop(nn.Module):
    def __init__(self, drop_type):
        super(Drop, self).__init__()
        if drop_type is None:
            self.drop = keep_origin
        elif drop_type == 'alpha':
            self.drop = nn.AlphaDropout(p=0.5)
        elif drop_type == 'dropout':
            self.drop = nn.Dropout3d(p=0.5)
        elif drop_type == 'drop_block':
            self.drop = DropBlock3D(drop_prob=0.2, block_size=2)
        else:
            raise NotImplementedError('{} not implemented'.format(drop_type))

    def forward(self, input):
        out = self.drop(input)
        return out
