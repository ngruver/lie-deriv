import os
import sys
import wandb
import numpy as np

sys.path.append("pytorch-image-models")
import timm.models
import argparse
import pandas as pd
from functools import partial

from loader import get_loaders

def numparams(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    # argparse
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
        "--train-split", metavar="NAME", default="train", help="dataset train split"
    )
    parser.add_argument(
        "--val-split",
        metavar="NAME",
        default="validation",
        help="dataset validation split",
    )
    parser.add_argument(
        "--expt",
        metavar="NAME",
        default="equivariance_metrics_10",
        help="experiment name",
    )
    parser.add_argument(
        "--modelname", metavar="NAME", default="resnet18", help="model name"
    )
    parser.add_argument(
        "--pretrained", metavar="NAME", default="True", help="use pretrained model"
    )
    args = parser.parse_args()

    wandb.init(project="metamixer", config=args)

    if not os.path.exists(args.expt):
        os.makedirs(args.expt)

    print(args.modelname)

    pretrained = True if args.pretrained == "True" else False
    model = getattr(timm.models, args.modelname)(pretrained=pretrained)
    model.eval()
    imagenet_train_loader, imagenet_test_loader = get_loaders(
        model,
        dataset="imagenet",
        data_dir="/imagenet",
        batch_size=8,
        num_train=100,
        num_val=100,
        args=args,
        train_split=args.train_split,
        val_split=args.val_split,
    )

    evaluated_metrics = []
    for key, loader in {
        "Imagenet_train": imagenet_train_loader,
        "Imagenet_test": imagenet_test_loader,
    }.items():
        metrics = eval_average_metrics_wstd(
            loader, partial(get_equivariance_metrics, model)
        )
        metrics["dataset"] = key
        metrics["model"] = args.modelname
        metrics["params"] = numparams(model)
        print(np.mean(metrics["acc"]))
        evaluated_metrics.append(metrics)

    cifar_train_loader, cifar_test_loader = get_loaders(
        model,
        dataset="torch/cifar100",
        data_dir="/scratch/nvg7279/cifar",
        batch_size=4,
        num_train=10000,
        num_val=5000,
        args=args,
        train_split=args.train_split,
        val_split=args.val_split,
    )

    svhn_train_loader, svhn_test_loader = get_loaders(
        model,
        dataset="torch/svhn",
        data_dir="/scratch/nvg7279/svhn",
        batch_size=4,
        num_train=10000,
        num_val=5000,
        args=args,
        train_split=args.train_split,
        val_split="test",
    )

    _, retinopathy_loader = get_loaders(
        model,
        dataset="tfds/diabetic_retinopathy_detection",
        data_dir="/scratch/nvg7279/tfds",
        batch_size=4,
        num_train=1e8,
        num_val=1e8,
        args=args,
        train_split="train",
        val_split="train",
    )

    _, histology_loader = get_loaders(
        model,
        dataset="tfds/colorectal_histology",
        data_dir="/scratch/nvg7279/tfds",
        batch_size=4,
        num_train=1e8,
        num_val=1e8,
        args=args,
        train_split="train",
        val_split="train",
    )

    for key, loader in {
        "cifar100": cifar_test_loader,
        "retinopathy": retinopathy_loader,
        "histology": histology_loader,
    }.items():
        metrics = eval_average_metrics_wstd(
            loader, partial(get_equivariance_metrics, model), max_mbs=(1000 // 4) + 1
        )
        metrics["dataset"] = key
        metrics["model"] = args.modelname
        metrics["params"] = numparams(model)
        evaluated_metrics.append(metrics)

    df = pd.concat(evaluated_metrics)
