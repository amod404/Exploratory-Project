# data/cifar100.py
# Auto-download: YES  (~170 MB, torchvision)
# Input: 32x32   Classes: 100   Batch: 128

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_cifar100_loaders(batch_size=128, num_workers=0, split_test=True,
                         fast_dev_mode=False):
    """
    Returns (train_loader, val_loader, test_loader).
    fast_dev_mode: uses 4500 train + 500 val images.
    Full mode    : uses 45000 train + 5000 val images.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762)),
    ])

    pin_mem = torch.cuda.is_available() and num_workers > 0

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_mem)

    train_full = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train)
    val_full = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_val)

    indices = list(range(len(train_full)))
    np.random.seed(42)
    np.random.shuffle(indices)

    if fast_dev_mode:
        train_idx, val_idx = indices[:4500], indices[4500:5000]
    else:
        train_idx, val_idx = indices[:45000], indices[45000:]

    test_ds = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_val)

    return (
        _loader(Subset(train_full, train_idx), shuffle=True),
        _loader(Subset(val_full,   val_idx),   shuffle=False),
        _loader(test_ds,                       shuffle=False),
    )