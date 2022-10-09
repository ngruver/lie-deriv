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

import lee.layerwise_lee as lee
import lee.layerwise_other as other_metrics
from lee.loader import get_loaders

import sys
sys.path.append('pytorch-image-models')
import timm

def convert_inplace_relu_to_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU())
        else:
            convert_inplace_relu_to_relu(child)

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
    print(args.transform)
    
    model = getattr(timm.models, args.modelname)(pretrained=True)
    
    convert_inplace_relu_to_relu(model)
    model = model.eval()
    model = model.to(torch.device("cuda"))

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
    if args.use_lee and (args.transform in lee_transforms):
        lee_model = copy.deepcopy(model)
        lee.apply_hooks(lee_model, args.transform)
        lee_metrics = get_layerwise(
            args, lee_model, loader, func=lee.compute_equivariance_attribution
        )
        
        lee_output_dir = os.path.join(args.output_dir, "lee_" + args.transform)
        os.makedirs(lee_output_dir, exist_ok=True)
        lee_metrics.to_csv(os.path.join(lee_output_dir, args.modelname + ".csv"))

    other_metrics_transforms = ["integer_translation","translation","rotation"]
    if (not args.use_lee) and (args.transform in other_metrics_transforms):
        other_metrics_model = copy.deepcopy(model)
        other_metrics.apply_hooks(other_metrics_model)
        func = partial(other_metrics.compute_equivariance_attribution, args.transform)
        other_metrics_results = get_layerwise(
            args, model, loader, func=func
        )

        other_metrics_output_dir = os.path.join(args.output_dir, "stylegan3_" + args.transform)
        os.makedirs(other_metrics_output_dir, exist_ok=True)
        results_fn = args.modelname + "_norm_sqrt" + ".csv"
        other_metrics_results.to_csv(os.path.join(other_metrics_output_dir, results_fn))


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
    parser.add_argument(
        "--use_lee", type=int, default=0, help="Use LEE (rather than metric not in limit)"
    )
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     main(args)
