import os
import tqdm
import argparse
import pandas as pd
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lie_derivs import *

class flag:
    pass

singleton = flag
singleton.compute_lie = True
singleton.op_counter = 0
singleton.fwd = True

# TODO: record call order
# TODO: if module is called 2nd time, create a copy and store separately
# change self inplace?
def store_inputs(lie_deriv_type, self, inputs, outputs):
    if not singleton.fwd or not singleton.compute_lie:
        return
    # if hasattr(self,'_lie_norm_sum'): return
    with torch.no_grad():
        singleton.compute_lie = False
        if not hasattr(self, "_lie_norm_sum"):
            self._lie_norm_sum = []
            self._lie_norm_sum_sq = []
            self._num_probes = []
            self._op_counter = []
            self._fwd_counter = 0
            self._lie_deriv_output = []
        if self._fwd_counter == len(self._lie_norm_sum):
            self._lie_norm_sum.append(0)
            self._lie_norm_sum_sq.append(0)
            self._num_probes.append(0)
            self._op_counter.append(singleton.op_counter)
            self._lie_deriv_output.append(0)
            singleton.op_counter += 1
            assert len(inputs) == 1
            (x,) = inputs
            x = x + torch.zeros_like(x)
            self._lie_deriv_output[self._fwd_counter] = lie_deriv_type(self, x)

        self._fwd_counter += 1
        singleton.compute_lie = True

def store_estimator(self, grad_input, grad_output):
    if singleton.compute_lie:
        with torch.no_grad():
            assert len(grad_output) == 1
            bs = grad_output[0].shape[0]
            self._fwd_counter -= 1  # reverse of forward ordering
            i = self._fwd_counter
            # if i!=0: return
            # print('bwd',self._fwd_counter)
            estimator = (
                (
                    (grad_output[0] * self._lie_deriv_output[i]).reshape(bs, -1).sum(-1)
                    ** 2
                )
                .cpu()
                .data.numpy()
            )
            self._lie_norm_sum[i] += estimator
            self._lie_norm_sum_sq[i] += estimator**2
            self._num_probes[i] += 1
            # print("finished bwd",self)


from timm.models.vision_transformer import Attention as A1
# from timm.models.vision_transformer_wconvs import Attention as A2
from timm.models.mlp_mixer import MixerBlock, Affine, SpatialGatingBlock
from timm.models.layers import PatchEmbed, Mlp, DropPath, BlurPool2d

# from timm.models.layers import FastAdaptiveAvgPool2d,AdaptiveAvgMaxPool2d
from timm.models.layers import GatherExcite, EvoNormBatch2d
from timm.models.senet import SEModule
from timm.models.efficientnet_blocks import SqueezeExcite
from timm.models.convit import MHSA, GPSA

leaflist = (
    A1,
    # A2,
    MixerBlock,
    Affine,
    SpatialGatingBlock,
    PatchEmbed,
    Mlp,
    DropPath,
    BlurPool2d,
)
# leaflist += (nn.AdaptiveAvgPool2d,nn.MaxPool2d,nn.AvgPool2d)
leaflist += (
    GatherExcite,
    EvoNormBatch2d,
    nn.BatchNorm2d,
    nn.BatchNorm1d,
    nn.LayerNorm,
    nn.GroupNorm,
    SEModule,
    SqueezeExcite,
)
leaflist += (MHSA, GPSA)

def is_leaf(m):
    return (not hasattr(m, "children") or not list(m.children())) or isinstance(
        m, leaflist
    )

def is_excluded(m):
    excluded_list = nn.Dropout
    return isinstance(m, excluded_list)

def selective_apply(m, fn):
    if is_leaf(m):
        if not is_excluded(m):
            fn(m)
    else:
        for c in m.children():
            selective_apply(c, fn)

def apply_hooks(model, lie_deriv_type):
    lie_deriv = {
        "translation": translation_lie_deriv,
        "rotation": rotation_lie_deriv,
        "hyper_rotation": hyperbolic_rotation_lie_deriv,
        "scale": scale_lie_deriv,
        "saturate": saturate_lie_deriv,
    }[lie_deriv_type]
    
    selective_apply(
        model, lambda m: m.register_forward_hook(partial(store_inputs, lie_deriv))
    )
    selective_apply(model, lambda m: m.register_backward_hook(store_estimator))


def reset(self):
    try:
        # del self._input
        del self._lie_norm_sum
        del self._lie_norm_sum_sq
        del self._num_probes
        del self._op_counter
        del self._bwd_counter
        del self._lie_deriv_output
    except AttributeError:
        pass


def reset2(self):
    self._lie_norm_sum = [0.0] * len(self._lie_norm_sum)
    self._lie_norm_sum_sq = [0.0] * len(self._lie_norm_sum)
    self._num_probes = [0.0] * len(self._lie_norm_sum)


def compute_equivariance_attribution(model, img_batch, num_probes=100):
    model_fn = lambda z: F.softmax(model(z), dim=1)
    all_errs = []
    order = []
    for j in range(num_probes):
        singleton.fwd = True
        y = model_fn(img_batch)
        z = torch.randn(img_batch.shape[0], 1000).to(torch.device("cuda"))
        loss = (z * y).sum()
        singleton.fwd = False
        loss.backward()
        model.zero_grad()
        singleton.op_counter = 0

        errs = {}
        for name, module in model.named_modules():
            if hasattr(module, "_lie_norm_sum"):
                for i in range(len(module._lie_norm_sum)):
                    assert module._num_probes[i] == 1

                    lie_norm = module._lie_norm_sum[i] / module._num_probes[i]
                    mod = module.__class__.__name__
                    errs[
                        (name + (f"{i}" if i else ""), mod, module._op_counter[i], i)
                    ] = lie_norm.item()

        errs = pd.Series(errs, index=pd.MultiIndex.from_tuples(errs.keys()), name=j)
        all_errs.append(errs)
        selective_apply(model, reset2)
    model.apply(reset)
    df = pd.DataFrame(all_errs)
    return df