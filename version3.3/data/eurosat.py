# data/eurosat.py
# =============================================================================
# EuroSAT: European Sentinel-2 Satellite Imagery Classification
# Auto-download: YES  (~90 MB, via torchvision.datasets.EuroSAT)
# Input: 64x64   Classes: 10   Batch: 64
#
# 27,000 RGB images at 64x64 across 10 land-use / land-cover classes:
#   AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
#   Pasture, PermanentCrop, Residential, River, SeaLake
#
# No official train/val/test split — we do a 70/15/15 stratified split.
# =============================================================================

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from utils.logger import get_logger

logger = get_logger("eurosat", logfile="logs/eurosat.log")

# EuroSAT RGB statistics (computed from the full dataset)
_MEAN = (0.3444, 0.3803, 0.4078)
_STD  = (0.2028, 0.1366, 0.1153)


def _stratified_split(dataset, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Returns (train_indices, val_indices, test_indices) using stratified sampling
    so each class is proportionally represented in every split.
    """
    np.random.seed(seed)

    # Group indices by class label
    class_indices = {}
    for idx in range(len(dataset)):
        # EuroSAT stores labels via .targets (list of ints)
        label = dataset.targets[idx]
        class_indices.setdefault(label, []).append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for indices in class_indices.values():
        indices = np.array(indices)
        np.random.shuffle(indices)
        n = len(indices)
        n_val  = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))
        test_idx.extend(indices[:n_test].tolist())
        val_idx.extend(indices[n_test:n_test + n_val].tolist())
        train_idx.extend(indices[n_test + n_val:].tolist())

    return train_idx, val_idx, test_idx


def get_eurosat_loaders(batch_size=64, num_workers=0, split_test=True,
                        fast_dev_mode=False):
    """
    Returns (train_loader, val_loader, test_loader).

    fast_dev_mode: uses 2000 train + 500 val images.
    Full mode    : full 27,000-image dataset with 70/15/15 stratified split.
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    pin_mem = torch.cuda.is_available() and num_workers > 0

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_mem)

    # torchvision.datasets.EuroSAT requires torchvision >= 0.11
    try:
        base_ds = torchvision.datasets.EuroSAT(
            root="./data", download=True, transform=transform_val)
    except AttributeError:
        raise RuntimeError(
            "torchvision.datasets.EuroSAT requires torchvision >= 0.11.\n"
            f"Installed version: {torchvision.__version__}\n"
            "Update with: pip install --upgrade torchvision"
        )

    # Check that .targets attribute exists (needed for stratified split)
    if not hasattr(base_ds, "targets"):
        # Older torchvision EuroSAT didn't expose .targets; build it manually
        base_ds.targets = [base_ds[i][1] for i in range(len(base_ds))]

    train_idx, val_idx, test_idx = _stratified_split(base_ds)

    if fast_dev_mode:
        np.random.seed(42)
        train_idx = np.random.choice(train_idx, size=min(2000, len(train_idx)),
                                     replace=False).tolist()
        val_idx   = np.random.choice(val_idx,   size=min(500,  len(val_idx)),
                                     replace=False).tolist()

    # Train split gets augmented transforms; val/test get clean transforms
    train_ds_aug = torchvision.datasets.EuroSAT(
        root="./data", download=True, transform=transform_train)

    logger.info("EuroSAT: train=%d  val=%d  test=%d",
                len(train_idx), len(val_idx), len(test_idx))

    return (
        _loader(Subset(train_ds_aug, train_idx), shuffle=True),
        _loader(Subset(base_ds,      val_idx),   shuffle=False),
        _loader(Subset(base_ds,      test_idx),  shuffle=False),
    )