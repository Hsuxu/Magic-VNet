# VNet Magic [Pytorch](https://pytorch.org/) Implements
Incluing VNet with [Attention module](https://arxiv.org/abs/1804.03999), VNet with [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) and [Atrous Spatial Pyramid Pooling](https://arxiv.org/abs/1706.05587)
1. A serious of [PyTorch] implementations of VNet with different changes for 3d volume segmentation.
(PS: This repository only contain the network module no data I/O including.  The network has not been tested on practical dataset due to computing resources.)

## Installation
- Install [PyTorch](https://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository
    ```
    git clone https://github.com/Hsuxu/vnet_attn.git
    cd vnet_attn
    ```

## Prerequisites
- Linux or maxOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Requirements

- numpy>=1.14.0
    ```
    pip install -r requirements.txt
    ```
- PyTorch>=0.4.0

---
## TODO
- Train the network on [PROMISE12](https://promise12.grand-challenge.org) 
