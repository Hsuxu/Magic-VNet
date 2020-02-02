import torch
import unittest
from magic_vnet.nestvnet import *


class NestVNetTest(unittest.TestCase):
    data = torch.rand((1, 1, 32, 32, 32))
    target_size = torch.Size([1, 1, 32, 32, 32])

    def test_nestvnet(self):
        model = NestVNet(1, 2)
        out = model(self.data)
        self.assertEqual(out.size(), self.target_size)

    def test_nestvnet_cse(self):
        model = NestVNet_CSE(1, 2)
        out = model(self.data)
        self.assertEqual(out.size(), self.target_size)

    def test_nestvnet_sse(self):
        model = NestVNet_SSE(1, 2)
        out = model(self.data)
        self.assertEqual(out.size(), self.target_size)

    def test_nestvbnet(self):
        model = NestVBNet(1, 2)
        out = model(self.data)
        self.assertEqual(out.size(), self.target_size)

    def test_sknestvnet(self):
        model = SK_NestVNet(1, 2, se_type=True)
        out = model(self.data)
        self.assertEqual(out.size(), self.target_size)


if __name__ == '__main__':
    unittest.main()
