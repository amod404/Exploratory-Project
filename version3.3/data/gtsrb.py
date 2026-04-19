# data/gtsrb.py
# =============================================================================
# GTSRB: German Traffic Sign Recognition Benchmark
# Auto-download: YES  (~300 MB, via torchvision.datasets.GTSRB)
# Input: 32x32   Classes: 43   Batch: 128
#
# Original images vary in size (30x30 to 266x232).
# We resize to 32x32 to keep the same input pipeline as CIFAR-based datasets.
#
# Train: 39,209 images   Test: 12,630 images
# No official val split — we split 10% from train for validation.
# =============================================================================

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from utils.logger import get_logger

logger = get_logger("gtsrb", logfile="logs/gtsrb.log")

# GTSRB ImageNet-style normalisation (computed from the training set)
_MEAN = (0.3403, 0.3121, 0.3214)
_STD  = (0.2724, 0.2608, 0.2669)


def get_gtsrb_loaders(batch_size=128, num_workers=0, split_test=True,
                      fast_dev_mode=False):
    """
    Returns (train_loader, val_loader, test_loader).

    fast_dev_mode: uses 4000 train + 800 val images.
    Full mode    : splits the full 39,209-image train set 90/10 into train/val.
    """
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    pin_mem = torch.cuda.is_available() and num_workers > 0

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_mem)

    # torchvision.datasets.GTSRB requires torchvision >= 0.12
    try:
        train_full = torchvision.datasets.GTSRB(
            root="./data", split="train", download=True,
            transform=transform_train)
        # Duplicate for val (different transform)
        val_full = torchvision.datasets.GTSRB(
            root="./data", split="train", download=True,
            transform=transform_val)
        test_ds = torchvision.datasets.GTSRB(
            root="./data", split="test", download=True,
            transform=transform_val)
    except AttributeError:
        raise RuntimeError(
            "torchvision.datasets.GTSRB requires torchvision >= 0.12.\n"
            f"Installed version: {torchvision.__version__}\n"
            "Update with: pip install --upgrade torchvision"
        )

    # Create 90/10 train/val split from the training set
    n = len(train_full)
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)

    if fast_dev_mode:
        train_idx = indices[:4000]
        val_idx   = indices[4000:4800]
    else:
        split = int(0.9 * n)
        train_idx = indices[:split]
        val_idx   = indices[split:]

    logger.info("GTSRB: train=%d  val=%d  test=%d",
                len(train_idx), len(val_idx), len(test_ds))

    return (
        _loader(Subset(train_full, train_idx), shuffle=True),
        _loader(Subset(val_full,   val_idx),   shuffle=False),
        _loader(test_ds,                       shuffle=False),
    )