# vnet_attention

A [PyTorch](https://pytorch.org/) implementation of VNet with attention mechanism for 3d volume segmentation.
(PS: This repository only contain the network no data I/O including.  The network has not been tested on practical dataset due to computing resources. This repository is my personal project so will not update on weekdays.)

## Installation
- Install [PyTorch](https://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository
    ```
    git clone https://github.com/Hsuxu/vnet_attention.git
    cd vnet_attention
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
- Add the network figure
- Train the network on [PROMISE12](https://promise12.grand-challenge.org)
- Add more flexible dataset 
