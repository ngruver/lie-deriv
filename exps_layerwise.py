import os
import tqdm
import copy
import argparse
import warnings
import pandas as pd
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import layerwise_lee as lee
import layerwise_discrete as discrete
from data.loader import get_loaders

import sys
sys.path.append('pytorch-image-models')
import timm

def convert_inplace_relu_to_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU())
        else:
            convert_inplace_relu_to_relu(child)

def prepare_model(model):
    convert_inplace_relu_to_relu(model)
    model.eval()
    model.to(torch.device("cuda"))

def get_layerwise(args, model, loader, func):
    errlist = []
    for idx, (x, _) in tqdm.tqdm(
        enumerate(loader), total=len(loader)
    ):
        if idx >= args.num_imgs:
            break
        
        img = x.to(torch.device("cuda"))
        errors = func(model, img, num_probes=args.num_probes)
        errors["img_idx"] = idx
        errlist.append(errors)

    df = pd.concat(errlist, axis=0)
    df["model"] = args.modelname
    return df

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(args.modelname)
    
    model = getattr(timm.models, args.modelname)(pretrained=True)
    prepare_model(model)

    _, loader = get_loaders(
        model,
        dataset="imagenet",
        data_dir="/imagenet",
        batch_size=1,
        num_train=args.num_imgs,
        num_val=args.num_imgs,
        args=args,
    )

    lee_transforms = ["translation","rotation","hyper_rotation","scale","saturate"]
    if args.transform in lee_transforms:
        lee_model = copy.deepcopy(model)
        lee.apply_hooks(lee_model, args.transform)
        lee_metrics = get_layerwise(
            args, lee_model, loader, func=lee.compute_equivariance_attribution
        )
        
        lee_output_dir = os.path.join(args.output_dir, "lee_" + args.transform)
        os.makedirs(lee_output_dir, exist_ok=True)
        lee_metrics.to_csv(os.path.join(lee_output_dir, args.modelname + ".csv"))

    discrete_transforms = ["integer_translation","translation","rotation"]
    if args.transform in discrete_transforms:
        discrete_model = copy.deepcopy(model)
        discrete.apply_hooks(discrete_model)
        func = partial(discrete.compute_equivariance_attribution, args.transform)
        discrete_metrics = get_layerwise(
            args, discrete_model, loader, func=func
        )

        discrete_output_dir = os.path.join(args.output_dir, "discrete_" + args.transform)
        os.makedirs(discrete_output_dir, exist_ok=True)    
        discrete_metrics.to_csv(os.path.join(discrete_output_dir, args.modelname + ".csv"))


def get_args_parser():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    parser.add_argument(
        '--output_dir', metavar='NAME', default='equivariance_metrics_cnns',help='experiment name'
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
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(args)