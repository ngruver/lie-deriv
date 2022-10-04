# The Lie Derivative for Learned Equivariance
<p align="center">
  <img src="/assets/title_figure.png" width=900>
</p>

# Installation instructions

Clone submodules and install requirements using
```bash
git clone --recurse-submodules https://github.com/ngruver/lie-deriv.git
cd lie-deriv
pip install -r requirements.txt
```

# Layer-wise equivariance experiments

The equivariance of individual models can be calculated using `exps_layerwise.py`, for example 
```bash
python exps_layerwise.py \
  --modelname=resnet50 \
  --output_dir=$HOME/lee_results \
  --num_imgs=20 \
  --num_probes=100 \
  --transform=translation 
```
Default models and transforms are available in our wandb [sweep configuration](https://github.com/ngruver/lie-deriv/blob/main/sweep_configs/e2e_configs.py)

# End-to-end equivariance experiments

```bash
python exps_e2e.py \
  --modelname=resnet50 \
  --output_dir=$HOME/lee_results \
  --num_datapoints=100 
```
We also include the wandb sweep configurations wandb [sweep configuration](https://github.com/ngruver/lie-deriv/blob/main/sweep_configs/e2e_configs.py) for our end-to-end equivariance experiments. 

# Plotting and Visualization

