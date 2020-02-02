import torch
import torch.nn as nn
from magic_vnet.blocks.skunit.skunit import SKConv3d, SK_Block

if __name__ == '__main__':
    down = torch.rand((1, 64, 32, 32, 32))
    # up = torch.rand((1, 16, 64, 64, 64))

    model = SK_Block(64, 64)
    out = model(down)
    print(out.shape)
