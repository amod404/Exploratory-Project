# data/cinic10.py
# =============================================================================
# CINIC-10: CIFAR-10 + ImageNet hybrid benchmark
# Auto-download: YES  (~426 MB from Zenodo)
# Input: 32x32   Classes: 10   Batch: 128
#
# CINIC-10 extends CIFAR-10 by mixing in ImageNet images downsampled to 32x32.
# It has 270,000 images total: 90,000 each in train/valid/test.
# Classes are the same 10 as CIFAR-10.
#
# Reference: Darlow et al., CINIC-10 Is Not ImageNet or CIFAR-10, 2018.
# =============================================================================

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from utils.logger import get_logger

logger = get_logger("cinic10", logfile="logs/cinic10.log")

_ROOT = "./data/CINIC-10"

# Zenodo public archive — stable academic URL
_DOWNLOAD_URL = "https://datashare.ed.ac.uk/download/DS_10283_3192.zip"
_FILENAME     = "CINIC-10.zip"

# Same class names and mean/std as CIFAR-10 (images are pixel-compatible)
_CINIC_MEAN = (0.47889522, 0.47227842, 0.43047404)
_CINIC_STD  = (0.24205776, 0.23828046, 0.25874835)


def _ensure_downloaded(root: str) -> None:
    """Download and extract CINIC-10 if not already present."""
    train_dir = os.path.join(root, "train")
    if os.path.isdir(train_dir):
        return  # already extracted

    logger.info("CINIC-10 not found at %s — downloading (~426 MB)...", root)
    os.makedirs("./data", exist_ok=True)

    try:
        from torchvision.datasets.utils import download_and_extract_archive
        download_and_extract_archive(
            url=_DOWNLOAD_URL,
            download_root="./data",
            extract_root="./data",
            filename=_FILENAME,
            remove_finished=False,  # keep zip in case re-extraction needed
        )
        logger.info("CINIC-10 downloaded and extracted to %s", root)
    except Exception as e:
        raise RuntimeError(
            f"Failed to auto-download CINIC-10: {e}\n\n"
            "Manual download:\n"
            "  1. Download: https://datashare.ed.ac.uk/download/DS_10283_3192.zip\n"
            "  2. Extract to ./data/ so you have ./data/CINIC-10/train/ etc.\n"
        ) from e


def get_cinic10_loaders(batch_size=128, num_workers=0, split_test=True,
                        fast_dev_mode=False, root=_ROOT):
    """
    Returns (train_loader, val_loader, test_loader).

    CINIC-10 has official train/valid/test splits (90k each).
    fast_dev_mode: uses 9000 train + 2000 valid samples.
    """
    _ensure_downloaded(root)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CINIC_MEAN, _CINIC_STD),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CINIC_MEAN, _CINIC_STD),
    ])

    pin_mem = torch.cuda.is_available() and num_workers > 0

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_mem)

    # CINIC-10 uses "valid" not "val"
    train_ds = torchvision.datasets.ImageFolder(
        os.path.join(root, "train"), transform=transform_train)
    val_ds   = torchvision.datasets.ImageFolder(
        os.path.join(root, "valid"), transform=transform_val)
    test_ds  = torchvision.datasets.ImageFolder(
        os.path.join(root, "test"),  transform=transform_val)

    if fast_dev_mode:
        np.random.seed(42)
        train_idx = np.random.choice(len(train_ds), size=9000, replace=False).tolist()
        val_idx   = np.random.choice(len(val_ds),   size=2000, replace=False).tolist()
        train_ds  = Subset(train_ds, train_idx)
        val_ds    = Subset(val_ds,   val_idx)

    return (
        _loader(train_ds, shuffle=True),
        _loader(val_ds,   shuffle=False),
        _loader(test_ds,  shuffle=False),
    )