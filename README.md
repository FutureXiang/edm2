# EDM2: Analyzing and Improving the Training Dynamics of Diffusion Models

This is a multi-gpu PyTorch implementation of the paper [Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696):
```bibtex
@article{karras2023analyzing,
  title={Analyzing and Improving the Training Dynamics of Diffusion Models},
  author={Karras, Tero and Aittala, Miika and Lehtinen, Jaakko and Hellsten, Janne and Aila, Timo and Laine, Samuli},
  journal={arXiv preprint arXiv:2312.02696},
  year={2023}
}
```
:exclamation: This repo only contains configs and experiments on small or medium scale datasets such as CIFAR-10/100 and Tiny-ImageNet. Full re-implementation on ImageNet-1k would be extremely expensive.

:fire: This repo contains implementations of `Config C`, `Config E` and the final `Config G` models. You can compare `block[C/E/G].py` and `unet[C/E/G].py` against each other to learn about the improvements proposed by the authors. :smile:

## Requirements
In addition to PyTorch environments, please install:
```sh
conda install pyyaml
pip install ema-pytorch tensorboard
```

## Usage
Use 4 GPUs to train unconditional Config C/E/G models on the CIFAR-100 dataset:
```sh
torchrun --nproc_per_node=4
  train.py  --config config/cifar100/C.yaml --use_amp
  train.py  --config config/cifar100/E.yaml --use_amp
  train.py  --config config/cifar100/G.yaml --use_amp
```

To generate 50000 images with different checkpoints, for example, run:
```sh
torchrun --nproc_per_node=4
  sample.py --config config/cifar100/C.yaml --use_amp --epoch 1000
  sample.py --config config/cifar100/E.yaml --use_amp --epoch 1600
  sample.py --config config/cifar100/G.yaml --use_amp --epoch 1999
```