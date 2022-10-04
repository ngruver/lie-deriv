# The Lie Derivative for Learned Equivariance
<p align="center">
  <img src="/assets/title_figure.png" width=900>
</p>

# Installation instructions

```bash
git clone --recurse-submodules https://github.com/ngruver/lie-deriv.git
cd lie-deriv
pip install -r requirements.txt
```

# Layer-wise equivariance experiments

```bash
python exps_e2e.py \
  --modelname=resnet50 \
  --output_dir=$HOME/lee_results \
  --num_imgs=20 \
  --num_probes=100 \
  --transform=translation 
```

# End-to-end equivariance experiments

```bash
python exps_e2e.py \
  --modelname=resnet50 \
  --output_dir=$HOME/lee_results \
  --num_datapoints=100 
```
