import tqdm
import pandas as pd
import random
from functools import partial
from typing import Callable

import torch
import torch.utils.data
import numpy as np

import sys
sys.path.append("pytorch-image-models")
from timm.data import create_dataset, resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.distributed_sampler import OrderedDistributedSampler

def _worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))

def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        crop_pct=None,
        collate_fn=None,
        pin_memory=False,
        tf_preprocessing=False,
        persistent_workers=True,
        worker_seeding='all',
        sampler=None,
):
    inner_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.dataset.Subset) \
                    else dataset

    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    inner_dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=False,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    return loader

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
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=1,
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
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=1,
        crop_pct=data_config["crop_pct"],
        pin_memory=False,
        no_aug=True,
    )

    return train_loader, eval_loader