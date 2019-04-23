# [VNet](https://arxiv.org/abs/1606.04797) Magic [PyTorch](https://pytorch.org/) Implements

This repository including some extension implements based original [VNet](https://arxiv.org/abs/1606.04797) network. (PS: The original VNet is not implemented)

## Modules
   
1. [Feature Pyramid Network](https://arxiv.org/abs/1612.03144)
2. [Attention module](https://arxiv.org/abs/1804.03999)
3. [Atrous Spatial Pyramid Pooling](https://arxiv.org/abs/1706.05587)
4. [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

## Useage
- Enter your project directory
```
cd [path to your projeaect]
```
- Clone this repository
```
git clone https://github.com/Hsuxu/Magic-VNet.git
cd Magic-VNet
```
- Install all requirements
```
pip install -r requirements.txt
```
- import all modules in your codes
```
from Magic_VNet import *
```

## TODO
- Train the network on [PROMISE12](https://promise12.grand-challenge.org) 

## Reference
- [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
- [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
