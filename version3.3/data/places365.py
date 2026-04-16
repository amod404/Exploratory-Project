# data/places365.py
# =============================================================================
# Places365 Scene Recognition
# Auto-download: YES (small 256px version, ~24 GB)
# Input: 224x224   Classes: 365   Batch: 32
#
# The "small" version uses 256x256 images which torchvision can auto-download.
# The standard (high-res) version is ~105 GB and requires manual download.
#
# NOTE: The first download takes a long time (~24 GB). Set FAST_DEV_MODE=True
# while testing to limit training to a small subset and save time.
# =============================================================================

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from utils.logger import get_logger

logger = get_logger("places365", logfile="logs/places365.log")

_ROOT = "./data/places365"


def get_places365_loaders(batch_size=32, num_workers=0, split_test=True,
                          fast_dev_mode=False):
    """
    Returns (train_loader, val_loader, test_loader).

    Uses small=True (256x256 pre-resized images) for auto-download.
    fast_dev_mode: 3000 train + 600 val images (first download still needed).
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    pin_mem = torch.cuda.is_available() and num_workers > 0

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_mem)

    try:
        train_ds = torchvision.datasets.Places365(
            root=_ROOT, split="train-standard", small=True,
            download=True, transform=transform_train)
        val_ds   = torchvision.datasets.Places365(
            root=_ROOT, split="val", small=True,
            download=True, transform=transform_val)
    except Exception as exc:
        raise RuntimeError(
            f"Places365 failed to load/download: {exc}\n\n"
            "If the download is stuck or fails, manual instructions:\n"
            "  http://places2.csail.mit.edu/index.html → Download\n"
            "  Extract to ./data/places365/\n"
            "Or set small=False in data/places365.py for the full dataset "
            "(not auto-downloadable, ~105 GB).\n"
        ) from exc

    if fast_dev_mode:
        np.random.seed(42)
        t_idx = np.random.choice(len(train_ds), size=min(3000, len(train_ds)),
                                 replace=False).tolist()
        v_idx = np.random.choice(len(val_ds),   size=min(600,  len(val_ds)),
                                 replace=False).tolist()
        train_ds = Subset(train_ds, t_idx)
        val_ds   = Subset(val_ds,   v_idx)

    logger.info("Places365: train=%d  val=%d",
                len(train_ds), len(val_ds))

    train_loader = _loader(train_ds, shuffle=True)
    val_loader   = _loader(val_ds,   shuffle=False)

    # Places365-standard val set has no separate test split → reuse val
    return train_loader, val_loader, val_loader