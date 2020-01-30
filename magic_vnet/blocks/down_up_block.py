import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicDownBlock(nn.Module):
    def forward(self, input):
        out = self.res_block(self.down_conv(input))
        out = self.drop(out)
        return out


class BasicUpBlock(nn.Module):
    def forward(self, input, skip):
        if self.upper == 'interpolation':
            target_size = skip.size()[2:]
            input = F.interpolate(input, size=target_size, mode='nearest')
        input = self.up_conv(input)
        out = torch.cat((input, skip),dim=1)
        out = self.drop(out)
        out = self.res_block(out)
        return out
