# [VNet](https://arxiv.org/abs/1606.04797) [PyTorch](https://pytorch.org/) Implements

This repository including some extension implements based original [VNet](https://arxiv.org/abs/1606.04797) network.

## Requirements
1. PyTorch>1.1.0
2. inplace_abn
please see the installation in [mapillary/inplace_abn](https://github.com/mapillary/inplace_abn#requirements) 
(PS: if you don't want the `inplace_abn` module just comment [line](https://github.com/Hsuxu/Magic-VNet/blob/75eda955a61a5875612cb9c9950ef5ca9bfea7a3/setup.py#L5) )

## Installation & Usage
- Clone this repository
```
git clone https://github.com/Hsuxu/Magic-VNet.git
cd Magic-VNet
python setup.py install
```
- import to your code
```python
from magic_vnet import VNet # network you need
```

## Reference
### Paper
- [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
- [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) 
- [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) 
- [Selective Kernel Networks](https://arxiv.org/abs/1903.06586)
- [Towards Stablizing Batch Statistics in Backward Propagation of Batch Normalization](https://arxiv.org/abs/2001.06838)
- [In-Place Activated BatchNorm for Memory-Optimized Training of DNNs](https://arxiv.org/abs/1712.02616)
    
### Code
- [ai-med/squeeze_and_excitation](https://github.com/ai-med/squeeze_and_excitation)
- [mapillary/inplace_abn](https://github.com/mapillary/inplace_abn)
- [megvii-model/MABN](https://github.com/megvii-model/MABN)
- [pppLang/SKNet](https://github.com/pppLang/SKNet)