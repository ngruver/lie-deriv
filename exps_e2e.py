import os
import gc
import wandb
import numpy as np
import argparse
import pandas as pd
from functools import partial

import sys
sys.path.append('pytorch-image-models')
import timm

from e2e_lee import get_equivariance_metrics as get_lee_metrics
from e2e_discrete import get_equivariance_metrics as get_discrete_metrics
from data.loader import get_loaders, eval_average_metrics_wstd

def numparams(model):
    return sum(p.numel() for p in model.parameters())

def get_metrics(args, key, loader, model, max_mbs=400):
    discrete_metrics = eval_average_metrics_wstd(
        loader, partial(get_discrete_metrics, model), max_mbs=max_mbs,
    )
    lee_metrics = eval_average_metrics_wstd(
        loader, partial(get_lee_metrics, model), max_mbs=max_mbs,
    )
    metrics = pd.concat([lee_metrics, discrete_metrics], axis=1)

    metrics["dataset"] = key
    metrics["model"] = args.modelname
    metrics["params"] = numparams(model)

    return metrics

def get_args_parser():
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('--output_dir', metavar='NAME', default='equivariance_metrics_cnns',help='experiment name')
    parser.add_argument('--modelname', metavar='NAME', default='resnet18', help='model name')
    parser.add_argument('--num_datapoints', type=int, default=60, help='use pretrained model')
    return parser

def main(args):
    wandb.init(project="LieDerivEquivariance", config=args)
    args.__dict__.update(wandb.config)

    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(args.modelname)

    model = getattr(timm.models, args.modelname)(pretrained=True)
    model.eval()

    evaluated_metrics = []

    imagenet_train_loader, imagenet_test_loader = get_loaders(
        model,
        dataset="imagenet",
        data_dir="/imagenet",
        batch_size=1,
        num_train=args.num_datapoints,
        num_val=args.num_datapoints,
        args=args,
        train_split='train',
        val_split='validation',
    )

    evaluated_metrics += [
        get_metrics(args, "Imagenet_train", imagenet_train_loader, model),
        get_metrics(args, "Imagenet_test", imagenet_test_loader, model)
    ]
    gc.collect()

    # _, cifar_test_loader = get_loaders(
    #     model,
    #     dataset="torch/cifar100",
    #     data_dir="/scratch/nvg7279/cifar",
    #     batch_size=1,
    #     num_train=args.num_datapoints,
    #     num_val=args.num_datapoints,
    #     args=args,
    #     train_split='train',
    #     val_split='validation',
    # )

    # evaluated_metrics += [get_metrics(args, "cifar100", cifar_test_loader, model, max_mbs=args.num_datapoints)]
    # gc.collect()

    # _, retinopathy_loader = get_loaders(
    #     model,
    #     dataset="tfds/diabetic_retinopathy_detection",
    #     data_dir="/scratch/nvg7279/tfds",
    #     batch_size=1,
    #     num_train=1e8,
    #     num_val=1e8,
    #     args=args,
    #     train_split="train",
    #     val_split="train",
    # )

    # evaluated_metrics += [get_metrics(args, "retinopathy", retinopathy_loader, model, max_mbs=args.num_datapoints)]
    # gc.collect()

    # _, histology_loader = get_loaders(
    #     model,
    #     dataset="tfds/colorectal_histology",
    #     data_dir="/scratch/nvg7279/tfds",
    #     batch_size=1,
    #     num_train=1e8,
    #     num_val=1e8,
    #     args=args,
    #     train_split="train",
    #     val_split="train",
    # )

    # evaluated_metrics += [get_metrics(args, "histology", histology_loader, model, max_mbs=args.num_datapoints)]
    # gc.collect()

    df = pd.concat(evaluated_metrics)
    df.to_csv(os.path.join(args.output_dir, args.modelname + ".csv"))

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
