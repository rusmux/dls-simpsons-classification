from __future__ import annotations

import os
from typing import Optional

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

num_workers = os.cpu_count()
num_workers = min(num_workers, 4) if num_workers else 0


def get_train_val_loader(
    dataset: ImageFolder,
    batch_size: int,
    train_size: Optional[float] = None,
    val_size: Optional[float] = None,
    weighted: bool = False,
) -> tuple[DataLoader, DataLoader]:

    targets = torch.tensor(dataset.targets)

    idxs = torch.arange(len(dataset))
    train_idxs, val_idxs, train_targets, _val_targets = train_test_split(
        idxs,
        targets,
        train_size=train_size,
        test_size=val_size,
        stratify=targets,
    )

    train_dataset = Subset(dataset, train_idxs)
    val_dataset = Subset(dataset, val_idxs)

    train_sampler = None
    if weighted:
        class_weights = 1 / torch.unique(targets, return_counts=True)[1]
        sample_weights = class_weights[train_targets]
        train_sampler = WeightedRandomSampler(sample_weights, len(train_idxs), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        sampler=train_sampler,
        num_workers=num_workers,  # type: ignore
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size,
        num_workers=num_workers,  # type: ignore
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(dataset: ImageFolder, batch_size: int) -> DataLoader:
    test_loader = DataLoader(
        dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True  # type: ignore
    )
    return test_loader
