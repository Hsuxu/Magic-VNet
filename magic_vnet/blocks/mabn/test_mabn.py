import os
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
import torch.nn as nn
from torch.optim import Adam

from mabn import MABN2d, MABN3d
from center_conv import CenConv2d, CenConv3d


class MABN3d_test(unittest.TestCase):
    data = torch.rand((2, 1, 32, 32, 32))
    target = torch.zeros((2, 32, 32, 32))
    target[:, 5:10, 5:10, 5:10] = 1
    target = target.long()
    model = nn.Sequential(nn.Conv3d(1, 16, kernel_size=3, padding=1),
                          MABN3d(16),
                          nn.ReLU(),
                          nn.Conv3d(16, 2, kernel_size=3, padding=1),
                          nn.Softmax(dim=1))

    def test_mabn_train(self):
        optim = Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.99))
        crti = nn.CrossEntropyLoss()
        for idx in range(5):
            out = self.model(self.data)
            loss = crti(out, self.target)
            loss.backward()
            optim.zero_grad()

            optim.step()
            print(loss.item())

    def test_mabn_inference(self):
        model = self.model.eval()
        with torch.no_grad():
            out = model(self.data)
            print(out.shape)


if __name__ == '__main__':
    unittest.main()
    # data = torch.rand((1, 16, 32, 32, 32))
    # bn = MABN3d(16)
    # out = bn(data)
    # print(out.shape)
    # data = torch.rand((1, 16, 32, 32))
    # bn = MABN2d(16)
    # out = bn(data)
    # print(out.shape)
