import torch
import unittest
from magic_vnet.vnet import *


class VNetTest(unittest.TestCase):
    data = torch.rand((1, 1, 32, 32, 32))
    target_size = torch.Size([1, 1, 32, 32, 32])

    def test_vnet(self):
        model = VNet(1, 2)
        out = model(self.data)
        self.assertEqual(out.size(), self.target_size)

    def test_vnet_cse(self):
        model = VNet_CSE(1, 2)
        out = model(self.data)
        self.assertEqual(out.size(), self.target_size)

    def test_vnet_sse(self):
        model = VNet_SSE(1, 2)
        out = model(self.data)
        self.assertEqual(out.size(), self.target_size)

    def test_vbnet(self):
        model = VBNet(1, 2)
        out = model(self.data)
        self.assertEqual(out.size(), self.target_size)

    def test_skvnet(self):
        model = SKVNet(1, 2)
        out = model(self.data)
        self.assertEqual(out.size(), self.target_size)


if __name__ == '__main__':
    unittest.main()
