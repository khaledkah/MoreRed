# MoreRed: Molecular Relaxation by Reverse Diffusion

<p align="center">
  <img src="https://github.com/khaledkah/MoreRed/assets/56682622/5f7a680e-7fd2-434e-b3a8-abc2aad6d39f" width="500" height="400">
</p>

MoreRed is built on top of [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master), an easily configurable and extendible library for constructing and training neural network models for atomistic systems like molecules. SchNetPack utilizes [PyTorch Lightning](https://www.pytorchlightning.ai/) for model building and [Hydra](https://hydra.cc/) for straightforward management of experimental configurations. While high level usage of the `morered` package to train and use the models described in the original paper does not require knowledge of its underlying dependencies, we recommend users familiarize themselves with Hydra to be able to customize their experimental configurations. Additionally, the tutorials and the documentation provided in [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master) can be helpful. Below, we explain how to use the `morered` package.

## Content

+ [Installation](/README.md##Installation)
+ [Usage](/README.md##Usage)
  + [Training](/README.md###Training)
  + [Molecular relaxation](/README.md###Molecular-relaxation)
  + [Molecular structure generation](/README.md###Molecular-structure-generation)
+ [How to cite](/README.md##How-to-cite)

## Installation
Requirements:
- python >= 3.8
- SchNetPack 2.0

You can install `morered` from the source code using pip, which will also install all its required dependencies including SchNetPack:

Download this repository. e.g. by cloning it using:
```
git clone git@github.com:khaledkah/MoreRed.git
cd MoreRed
```
We recommend creating a new Python environment or using conda to avoid incompatibilities with previously installed packages. E.g. if using conda:
```
conda create -n morered python=3.12
conda activate morered
```
Now to install the package, inside the folder `MoreRed` run:
```
pip install .
```
## Usage
The human-readable and customizable YAML configuration files under `/src/morered/configs` are all you need to train and run customizable experiments with `morered`. They follow the configuration structure used in [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master). Here, we explain how to train and use the different models. Besides, under the folder `notebooks` we provide step-by-step Jupyter notebooks explaining the building blocks of MoreRed and how to use the different trained models.

### Training

#### MoreRed-JT
You can train the `MoreRed-JT` model with the default configuration by simply running:
```
pip install .
```
### Molecular relaxation

### Molecular structure generation

## How to cite
