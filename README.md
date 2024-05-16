# U-MLP

Pytorch Code base for “**U-MLP: MLP-based Ultralight Refinement Network for Multimodal Medical Image Segmentation**”,The full code will be published after the article is accepted! Thank you for your attention

## Introduction

The convolutional neural network (CNN) and Transformer play an important role in computer-aided diagnosis and intelligent medicine. However, CNN cannot obtain long-range dependence, and Transformer has the shortcomings in computational complexity and a large number of parameters. Recently, compared with CNN and Transformer, the Multi-Layer Perceptron (MLP)-based medical image processing network can achieve higher accuracy with smaller computational and parametric quantities. Hence, in this work, we propose an encoder-decoder network, U-MLP, based on the ReMLP block. 

![1-s2 (1)](https://github.com/xielaobanyy/U-MLP/assets/92131703/5c109e41-cc13-460d-858f-1df765e4807d)


## Using the code:

The code is stable while using Python 3.8.0, CUDA >=11.1

- Clone this repository:
```bash
git clone https://github.com/xielaobanyy/U-MLP
cd U-MLP
```

To install all the dependencies :

```bash
conda env create U-MLP python==3.8.0
conda activate U-MLP
pip install -r requirements.txt
```

## Datasets

1) ISIC-2018 - [Link](https://challenge.isic-archive.com/data/)
2) MSD-Spleen - [Link](http://medicaldecathlon.com/)
2) MSD-heart - [Link](http://medicaldecathlon.com/)
2) CHAOS - [Link](https://zenodo.org/record/3431873)

## Data Format

Make sure to put the files as the following structure (e.g. the number of classes is 2):

```
inputs
└── <dataset name>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 001.png
        |   ├── 002.png
        |   ├── 003.png
        |   ├── ...
```

For binary segmentation problems, just use folder 0.

## Training and Validation

1. Train the model.
```
python train.py --dataset <dataset name> --arch resmlp_12
```
2. Evaluate.
```
python val.py --name <exp name>
```

### Acknowledgements：
This code-base uses certain code-blocks and helper functions from UNeXt [Link](https://github.com/jeya-maria-jose/UNeXt-pytorch).

### Citation:
```
@article{Shuo Gao U-MLP,
  title={U-MLP: MLP-based Ultralight Refinement Network for Multimodal Medical Image Segmentation},
  author={Shuo Gao, Wenhui Yang, Menglei Xu, Hao Zhang, Hong Yu, Airong Qian, Wenjuan Zhang},
  journal={Computers in Biology and Medicine},
  DOI={https://doi.org/10.1016/j.compbiomed.2023.107460}
}
```
