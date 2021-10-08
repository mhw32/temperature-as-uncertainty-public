# Temperature as Uncertainty

A PyTorch implementation of *Temperature as Unceratainty* (ArXiv link to come).

## Abstract 

Contrastive learning has demonstrated great capability to learn representations without annotations, even outperforming supervised baselines. However, it still lacks important properties useful for real-world application, one of which is uncertainty. In this paper, we propose a simple way to generate uncertainty scores for many contrastive methods by re-purposing temperature, a mysterious hyperparameter used for scaling. By observing that temperature controls how sensitive the objective is to specific embedding locations, we aim to learn temperature as an input-dependent variable, treating it as a measure of embedding confidence. We call this approach "Temperature as Uncertainty", or TaU. Through experiments, we demonstrate that TaU is useful for out-of-distribution detection, while remaining competitive with benchmarks on linear evaluation.  Moreover, we show that TaU can be learned on top of pretrained models, enabling uncertainty scores to be generated post-hoc with popular off-the-shelf models. In summary, TaU is a simple yet versatile method for generating uncertainties for contrastive learning.

### Main Intuition

Learn an extra mapping from input to temperature. Treat this temperature as an uncertainty score. Lower temperature represents more sensitivity to small changes to embedding location whereas higher temperature represents less sensitivity. One can use this simple extension to the contrastive framework for OOD detection.

## Setup/Installation

We use Python 3, PyTorch 1.7.1, PyTorch Lightning 1.1.8, and a conda environment. Consider a variation of the commands below:

```
conda create -n tau python=3 anaconda
conda activate tau
conda install pytorch=1.7.1 torchvision -c pytorch
pip install pytorch-lightning=1.1.8
pip install tqdm dotmap
```

## Data

This repo only contains public data, most of which is found from `torchvision`. For pretrained uncertainty experiments, we utilize [COCO](https://cocodataset.org/#home), [ImageNet](https://www.image-net.org/), [LSUN](https://www.yf.io/p/lsun), and [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), all of which is public as well but must be downloaded seperately. 

## Usage

For every fresh terminal instance, you should run
```
source init_env.sh
```
to add the correct paths to `sys.path` before running anything else.

The primary script is found in the `scripts/run.py` file. It is used to run pretraining and OOD, linear evaluation experiments. You must supply it a configuration file, for which many templates are in the `configs/` folder. These configuration files are not complete, you must supply a experiment base directory (`exp_base`) to point to where in your filesystem model checkpoints will go. You must also supply a `root` directory where data should be downloaded to.

Example usage:

```
python scripts/run.py <CONFIG_FILE>
```

For linear evaluation, in the config file, you must provide the exp_dir and checkpoint_name (the file containing the epoch name) for the pretrained model.

## Citation

If you find this useful for your research, please cite:

TODO.
