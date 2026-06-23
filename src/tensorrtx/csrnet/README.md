# csrnet

## Overview

The Pytorch implementation is [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch).

This repo is a TensorRT implementation of CSRNet.

PartA model download url: https://drive.google.com/file/d/1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH/view

paper : [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062)

## Usage

1. download the pretrained model file and put it at `tensorrtx/models/PartAmodel_best.pth.tar` (skip if you already have it).

2. generate `csrnet.wts` from `PartAmodel_best.pth.tar`

```bash
pushd tensorrtx/csrnet
python3 gen_wts.py
popd
```

3. build C++ code

```bash
pushd tensorrtx/csrnet
cmake -S . -B build -G Ninja --fresh
cmake --build build
popd
```

4. serialize to `tensorrtx/models/csrnet.engine`

```bash
pushd tensorrtx/csrnet
./build/csrnet -s
popd
```

5. run inference (reads `tensorrtx/assets/IMG_1.jpg`, writes `tensorrtx/assets/csrnet_output.jpg`)

```bash
pushd tensorrtx/csrnet
./build/csrnet -d
popd
```

# result

output looks like:

```bash
...
Execution time: 62956us
0.000543074 0.00035309 0.00351256 0.00194812 0.00201269 0.000398191 -0.000205946 -0.000301642 0.000487901 -9.49481e-05
====
Execution time: 62431us
0.000543074 0.00035309 0.00351256 0.00194812 0.00201269 0.000398191 -0.000205946 -0.000301642 0.000487901 -9.49481e-05
====
approximate people num: 21
```

you can also check the output image in `tensorrtx/assets/csrnet_output_tensorrt.jpg`:

![output](../assets/csrnet_output_tensorrt.jpg)
