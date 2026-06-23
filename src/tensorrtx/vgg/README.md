# VGG

## Introduction

This is a TensorRT implementation of torchvision VGG variants from [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf). It supports the common torchvision VGG11/VGG13/VGG16/VGG19 models and their batch-normalized variants.

VGG's architecture is simple, just some conv, relu, maxpool, and fc layers, e.g., for VGG11:

```bash
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): ReLU(inplace=True)
    (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (14): ReLU(inplace=True)
    (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)
    (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (19): ReLU(inplace=True)
    (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

## Usage

1. Set the cache locations used by PyTorch/Hugging Face. The export script only reads these paths from environment variables.

```bash
export TORCH_HOME=/path/to/torch-cache
export HF_HOME=/path/to/hf-cache
```

2. Run `gen_wts.py` to generate `.wts` files. No inference is run during export.

```bash
python gen_wts.py --model vgg11
```

Omit `--model` to export all supported variants, or repeat it for multiple variants.

3. Supported model names:

| model name | torchvision config | batch norm |
| ---------- | ------------------ | ---------- |
| vgg11      | A                  | no         |
| vgg11_bn   | A                  | yes        |
| vgg13      | B                  | no         |
| vgg13_bn   | B                  | yes        |
| vgg16      | D                  | no         |
| vgg16_bn   | D                  | yes        |
| vgg19      | E                  | no         |
| vgg19_bn   | E                  | yes        |

4. build C++ code

```bash
pushd tensorrtx/vgg
cmake -S . -B build -G Ninja --fresh
cmake --build build
```

5. serialize wts model to engine file

```bash
./build/vgg -s
```

Pass a model name to serialize a non-default variant:

```bash
./build/vgg -s vgg16_bn
```

6. run inference

```bash
./build/vgg -d vgg11
```
