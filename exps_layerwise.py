import torch
import torch.nn.functional as F
import torch.nn as nn

# need to pip install ml_collections

import os
import sys
import tqdm

sys.path.append("pytorch-image-models")
import timm.models

import argparse
import pandas as pd
from functools import partial

from loader import get_loaders

def convert_inplace_relu_to_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU())
        else:
            convert_inplace_relu_to_relu(child)


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
        # print('fwd',self._fwd_counter)
        singleton.compute_lie = True
        # print("finished fwd",self)


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
from timm.models.vision_transformer_wconvs import Attention as A2
from timm.models.mlp_mixer import MixerBlock, Affine, SpatialGatingBlock
from timm.models.layers import PatchEmbed, Mlp, DropPath, BlurPool2d

# from timm.models.layers import FastAdaptiveAvgPool2d,AdaptiveAvgMaxPool2d
from timm.models.layers import GatherExcite, EvoNormBatch2d
from timm.models.senet import SEModule
from timm.models.efficientnet_blocks import SqueezeExcite
from timm.models.convit import MHSA, GPSA

leaflist = (
    A1,
    A2,
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


# def excluded(module):
#     if isinstance(module,):
#         return False
#     if list(module.children()):
#         return True # not a leaf
#     if isinstance(module, nn.Dropout):
#         return True
#     return False

# def register_store_inputs(module):
#     module.register_forward_hook(store_inputs)
# def register_store_estimator(module):
#     module.register_backward_hook(store_estimator)
#     #module.register_backward_hook(store_estimator) #which to use?
def apply_hooks(model, lie_deriv_type):
    selective_apply(
        model, lambda m: m.register_forward_hook(partial(store_inputs, lie_deriv_type))
    )
    selective_apply(model, lambda m: m.register_backward_hook(store_estimator))


def compute_equivariance_attribution(model, img_batch, num_probes=100):
    BS = img_batch.shape[0]
    model_fn = lambda z: F.softmax(model(z), dim=1)
    all_errs = []
    order = []
    for j in range(num_probes):
        singleton.fwd = True
        y = model_fn(img_batch)
        z = torch.randn(img_batch.shape[0], 1000).to(device)
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

        # errs = pd.Series(errs)
        errs = pd.Series(errs, index=pd.MultiIndex.from_tuples(errs.keys()), name=j)
        all_errs.append(errs)
        selective_apply(model, reset2)
    model.apply(reset)
    df = pd.DataFrame(all_errs)
    return df


if __name__ == "__main__":
    config_parser = parser = argparse.ArgumentParser(
        description="Training Config", add_help=False
    )
    parser.add_argument(
        "-c",
        "--config",
        default="",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # Dataset / Model parameters
    parser.add_argument(
        "--data_dir", metavar="DIR", default="/imagenet", help="path to dataset"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        metavar="NAME",
        default="",
        help="dataset type (default: ImageFolder/ImageTar if empty)",
    )
    parser.add_argument(
        "--train-split",
        metavar="NAME",
        default="/imagenet/train",
        help="dataset train split",
    )
    parser.add_argument(
        "--val-split",
        metavar="NAME",
        default="/imagenet/val",
        help="dataset validation spli",
    )
    parser.add_argument(
        "--expt", metavar="NAME", default="layerwise_metrics", help="experiment name"
    )
    parser.add_argument(
        "--modelname", metavar="NAME", default="resnetblur18", help="model name"
    )
    parser.add_argument(
        "--num_imgs", type=int, default=20, help="Number of images to evaluate over"
    )
    parser.add_argument(
        "--num_probes",
        type=int,
        default=100,
        help="Number of probes to use in the estimator",
    )
    parser.add_argument(
        "--transform",
        metavar="NAME",
        default="translation",
        help="translation or rotation",
    )
    args = parser.parse_args()
    # if not os.path.exists(args.expt):
    #     os.makedirs(args.expt)
    # print(args.modelname)
    # modelname = 'resnetblur50'
    pretrained = True
    model = getattr(timm.models, args.modelname)(pretrained=pretrained)
    convert_inplace_relu_to_relu(model)
    # model = nn.Sequential(*get_children(model))
    # print(model)
    model.eval()
    device = torch.device("cuda")
    model.to(device)
    # model.to('cpu')#cuda')

    # selectively apply hook to model
    # model.apply(register_store_inputs)
    # model.apply(register_store_estimator)lS
    lie_deriv_type = {
        "translation": translation_lie_deriv,
        "rotation": rotation_lie_deriv,
        "hyper_rotation": hyperbolic_rotation_lie_deriv,
        "scale": scale_lie_deriv,
        "saturate": saturate_lie_deriv,
    }[args.transform]

    apply_hooks(model, lie_deriv_type)

    BS = 1
    imagenet_train_loader, imagenet_test_loader = get_loaders(
        model,
        dataset="imagenet",
        data_dir="/imagenet",
        batch_size=BS,
        num_train=args.num_imgs,
        num_val=args.num_imgs,
        args=args,
    )
    errlist = []
    for idx, (x, y) in tqdm.tqdm(
        enumerate(imagenet_test_loader), total=len(imagenet_test_loader)
    ):
        if idx >= args.num_imgs:
            break
        img = x.to(device)  # x[0][None].to(device)
        errors = compute_equivariance_attribution(
            model, img, num_probes=args.num_probes
        )
        # for k,(v,err,mod,count,*rest) in errors.items():
        #     #(avg_lie_norm,avg_probe_stderr,mod,module._op_counter[i],module._num_probes[i])
        #     print(f"{k:<25} |Lf|^2 contribution: {np.sqrt(v):.4f} ± {np.sqrt(err):.4f}. {mod} {count}")
        # print(f"Sum of |Lf|^2 contributions: {sum([np.sqrt(v) for _,(v,*rest) in errors.items()]):.4f}")
        singleton.compute_lie = False
        L_all = (
            (translation_lie_deriv(lambda z: F.softmax(model(z), dim=1), img) ** 2)
            .reshape(BS, -1)
            .sum()
            .cpu()
            .data.numpy()
        )
        errors["L_all"] = L_all
        errors["img_idx"] = idx
        singleton.compute_lie = True
        # print(f"Full |L|^2: {np.sqrt(L_all):.4f}")
        errlist.append(errors)

    df = pd.concat(errlist, axis=0)
    df["model"] = args.modelname
    os.makedirs(args.expt + "_" + args.transform, exist_ok=True)
    df.to_csv(os.path.join(args.expt + "_" + args.transform, args.modelname + ".csv"))
