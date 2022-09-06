import torch
import tqdm
import pandas as pd

import sys
sys.path.append("pytorch-image-models")
from timm.data import create_dataset, create_loader, resolve_data_config

def eval_average_metrics_wstd(loader, metrics, max_mbs=None):
    total = len(loader) if max_mbs is None else min(max_mbs, len(loader))
    dfs = []
    with torch.no_grad():
        for idx, minibatch in tqdm.tqdm(enumerate(loader), total=total):
            dfs.append(metrics(minibatch))
            if max_mbs is not None and idx >= max_mbs:
                break
    df = pd.concat(dfs)
    return df

def get_loaders(
    model,
    dataset,
    data_dir,
    batch_size,
    num_train,
    num_val,
    args,
    train_split="train",
    val_split="val",
):

    dataset_train = create_dataset(
        dataset,
        root=data_dir,
        split=train_split,
        is_training=False,
        batch_size=batch_size,
    )
    if num_train < len(dataset_train):
        dataset_train, _ = torch.utils.data.random_split(
            dataset_train,
            [num_train, len(dataset_train) - num_train],
            generator=torch.Generator().manual_seed(42),
        )

    dataset_eval = create_dataset(
        dataset,
        root=data_dir,
        split=val_split,
        is_training=False,
        batch_size=batch_size,
    )
    if num_val < len(dataset_eval):
        dataset_eval, _ = torch.utils.data.random_split(
            dataset_eval,
            [num_val, len(dataset_eval) - num_val],
            generator=torch.Generator().manual_seed(42),
        )

    data_config = resolve_data_config(vars(args), model=model, verbose=True)

    print(data_config)

    train_loader = create_loader(
        dataset_train,
        input_size=data_config["input_size"],
        batch_size=batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=1,
        distributed=False,
        crop_pct=data_config["crop_pct"],
        pin_memory=False,
        no_aug=True,
        hflip=0.0,
        color_jitter=0.0,
    )

    eval_loader = create_loader(
        dataset_eval,
        input_size=data_config["input_size"],
        batch_size=batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=1,
        distributed=False,
        crop_pct=data_config["crop_pct"],
        pin_memory=False,
        no_aug=True,
    )

    return train_loader, eval_loader