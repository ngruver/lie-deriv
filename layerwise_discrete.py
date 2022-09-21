import os
import pandas as pd
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F

import sys
sys.path.append("stylegan3")
from metrics.equivariance import (
    apply_integer_translation,
    apply_fractional_translation,
    apply_fractional_rotation,
    apply_fractional_pseudo_rotation,
)

from layerwise_lee import selective_apply
from data.transforms import img_like

def EQ_T(module, img, model_out, translate_max=0.125):
    t = (torch.rand(2, device='cuda') * 2 - 1) * translate_max
    t = (t * img.shape[-2] * img.shape[-2]).round() / (img.shape[-2] * img.shape[-1])
    ref, _ = apply_integer_translation(img, t[0], t[1])
    t_model_out = module(ref)
    t_model_out, out_mask = apply_integer_translation(t_model_out, -t[0], -t[1]) 
    squared_err = (t_model_out - model_out).square()
    denom = out_mask.sum() if out_mask.sum() > 0 else 1.0
    mse = (squared_err * out_mask).sum() / denom
    # psnr = np.log10(2) * 20 - mse.log10() * 10
    return mse

def EQ_T_frac(module, img, model_out, translate_max=0.125):
    t = (torch.rand(2, device='cuda') * 2 - 1) * translate_max
    ref, _ = apply_fractional_translation(img, t[0], t[1])
    t_model_out = module(ref)
    t_model_out, out_mask = apply_fractional_translation(t_model_out, -t[0], -t[1]) 
    squared_err = (t_model_out - model_out).square()
    denom = out_mask.sum() if out_mask.sum() > 0 else 1.0
    mse = (squared_err * out_mask).sum() / denom
    # psnr = np.log10(2) * 20 - mse.log10() * 10
    return mse

def EQ_R(module, img, model_out, rotate_max=1.0):
    angle = (torch.rand([], device='cuda') * 2 - 1) * (rotate_max * np.pi)
    ref, _ = apply_fractional_rotation(img, angle)
    t_model_out = module(ref)
    t_model_out, out_mask = apply_fractional_rotation(t_model_out, -angle) 
    # model_out, pseudo_mask = apply_fractional_pseudo_rotation(model_out, -angle)
    squared_err = (t_model_out - model_out).square()
    denom = out_mask.sum() if out_mask.sum() > 0 else 1.0
    mse = (squared_err * out_mask).sum() / denom
    # psnr = np.log10(2) * 20 - mse.log10() * 10
    return mse


def store_inputs(self, inputs, outputs):
    self._cached_input = inputs
    self._cached_output = outputs

def apply_hooks(model):    
    selective_apply(
        model, lambda m: m.register_forward_hook(store_inputs)
    )

def reset(self):
    try:
        del self._cached_input
    except AttributeError:
        pass

def compute_equivariance_attribution(transform, model, img_batch, num_probes=20):
    equiv_metric = {
        "integer_translation": EQ_T,
        "translation": EQ_T_frac,
        "rotation": EQ_R,
    }[transform]
    
    model_fn = lambda z: F.softmax(model(z), dim=1)
    with torch.no_grad():
        model_fn(img_batch)
    
    all_errs = []
    for j in range(num_probes):
        errs = {}
        name_counter = defaultdict(int)
        for i, (name, module) in enumerate(model.named_modules()):
            name_counter[name] += 1

            if not hasattr(module, "_cached_input") or \
            not hasattr(module, "_cached_output"):
                continue

            model_in = module._cached_input
            model_in = model_in[0] if isinstance(model_in, tuple) else model_in
            model_out = module._cached_output
            model_out = model_out[0] if isinstance(model_out, tuple) else model_out

            if not img_like(model_in.shape) or not img_like(model_out.shape):
                equiv_err = 0.0
            else:
                with torch.no_grad():
                    equiv_err = equiv_metric(module, model_in, model_out).cpu().data.numpy()

            num_tag = f"{name_counter[name]}" if name_counter[name] else ""
            mod = module.__class__.__name__
            errs[
                (name + (num_tag), mod, i, name_counter[name])
            ] = equiv_err

        errs = pd.Series(errs, index=pd.MultiIndex.from_tuples(errs.keys()), name=j)
        all_errs.append(errs)

    model.apply(reset)
    df = pd.DataFrame(all_errs)
    return df