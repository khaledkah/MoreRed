# MoreRed: Molecular Relaxation by Reverse Diffusion

<table align="center", border=0>
  <tr>
    <td rowspan="2">
      <img src="https://github.com/khaledkah/MoreRed/assets/56682622/5f7a680e-7fd2-434e-b3a8-abc2aad6d39f" width="550" height="420">
    </td>
    <td>
      <img src="https://github.com/khaledkah/MoreRed/assets/56682622/a02032ba-a3a2-4b20-9658-faada1cbdd73" width="300" height="200">
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/khaledkah/MoreRed/assets/56682622/dc18a881-8abc-48c8-a704-e10dc528998c" width="300" height="200">
    </td>
  </tr>
</table>

MoreRed is built on top of [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master), an easily configurable and extendible library for constructing and training neural network models for atomistic systems like molecules. SchNetPack utilizes [PyTorch Lightning](https://www.pytorchlightning.ai/) for model building and [Hydra](https://hydra.cc/) for straightforward management of experimental configurations. While high level usage of the `morered` package to train and use the models described in the original paper does not require knowledge of its underlying dependencies, we recommend users familiarize themselves with Hydra to be able to customize their experimental configurations. Additionally, the tutorials and the documentation provided in [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master) can be helpful. Below, we explain how to use the `morered` package.

**_NOTE: while the current documentation in the README file and the source code should be sufficient to easily use the package, we are continually enhancing it._**

#### Content

+ [Installation](/README.md##Installation)
+ [Training](/README.md##Training)
+ [Molecular relaxation](/README.md##Molecular-relaxation)
+ [Molecular structure generation](/README.md##Molecular-structure-generation)
+ [Tutorials](/README.md##Tutorials)
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

## Training
The human-readable and customizable YAML configuration files under `src/morered/configs` are all you need to train and run customizable experiments with `morered`. They follow the configuration structure used in [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master). Here, we explain how to train and use the different models. Besides, under the folder `notebooks` we provide step-by-step Jupyter notebooks explaining the building blocks of MoreRed and how to use the different trained models.

Installing `morered` using pip adds the new CLI command `mrdtrain`, which can be used to train the different models by running the command:
```
mrdtrain experiment=<my-experiemnt>
```
where `<my-experiment>` specifies the experimental configurations to be used. It can either be one of the pre-installed experiments within the package, under `src/morered/configs/experiments`, or a path to a new YAML file created by the user. Detailed instructions on creating custom configurations can be found in the documentation of [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master).

In the original paper, three variants of MoreRed were introduced:

#### MoreRed-JT:
You can train the `MoreRed-JT` variant on QM7-X with the default configuration by simply running:
```
mrdtrain experiment=vp_gauss_morered_jt
```

#### MoreRed-AS/ITP:
Both variants, `MoreRed-AS` and `MoreRed-ITP`, require a separately trained time predictor and a noise predictor. The noise predictor here is also the usual DDPM model and can be trained using:
```
mrdtrain experiment=vp_gauss_ddpm
```
The time predictor can be trainined by running:
```
mrdtrain experiment=vp_gauss_time_predictor
```

#### Train on QM9
To train the models on QM9 instead of QM7-X you can append the suffix `_qm9` to the experiment name, for instance by running:
```
mrdtrain experiment=vp_gauss_morered_jt_qm9
```
Otherwise you can use the CLI to overwrite the Hydra configurations of the data set by running:
```
mrdtrain experiment=vp_gauss_morered_jt data=qm9_filtered
```
More about overwriting configurations in the CLI can be found in the [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master) documentation. 

## Molecular relaxation
The notebook `notebooks/denoising_tutorial.ipynb` explains how the trained models can be used for denoising.

## Molecular structure generation

## Tutorials
Under `notebooks`, we provide different tutorial in the form of Jupyter notebooks:
  - `diffusion_tutorial.ipynb`: explains how to use the diffusion processes implemented in `morered`.

## How to cite
