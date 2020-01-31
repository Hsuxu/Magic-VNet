import torch
import torch.nn as nn
from magic_vnet.blocks import DownBlock, UpBlock

if __name__ == '__main__':
    down = torch.rand((1, 32, 32, 32, 32))
    up = torch.rand((1, 16, 64, 64, 64))

    model = UpBlock(32, 16, 32, upper='interpolation')
    out = model(down, up)
    print(out.shape)
