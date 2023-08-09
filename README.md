# U-MLP

Pytorch Code base for “**U-MLP: MLP-based Ultralight Refinement Network for Multimodal Medical Image Segmentation**”,The full code will be published after the article is accepted! Thank you for your attention

## Introduction

The convolutional neural network (CNN) and Transformer play an important role in computer-aided diagnosis and intelligent medicine. However, CNN cannot obtain long-range dependence, and Transformer has the shortcomings in computational complexity and a large number of parameters. Recently, compared with CNN and Transformer, the Multi-Layer Perceptron (MLP)-based medical image processing network can achieve higher accuracy with smaller computational and parametric quantities. Hence, in this work, we propose an encoder-decoder network, U-MLP, based on the ReMLP block. 

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
python train.py --dataset <dataset name> --arch UMLP
```
2. Evaluate.
```
python val.py --name <exp name>
```

### Acknowledgements:

This code-base uses certain code-blocks and helper functions from [UNext]([jeya-maria-jose/UNeXt-pytorch：“UNeXt：基于 MLP 的快速医学图像分割网络”的官方 Pytorch 代码库，MICCAI 2022 (github.com)](https://github.com/jeya-maria-jose/UNeXt-pytorch)),  and Multi-class segmentation mainly refers to ([WZMIAOMIAO/deep-learning-for-image-processing: deep learning for image processing including classification and object-detection etc. (github.com)](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)). Thanks for their great works.

### Citation:
```
@article{Shuo Gao U-MLP,
  title={U-MLP: MLP-based Ultralight Refinement Network for Multimodal Medical Image Segmentation},
  author={Shuo Gao, Wenhui Yang and Menglei Xu},
}
```
