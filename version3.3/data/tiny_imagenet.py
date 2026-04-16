# data/tiny_imagenet.py
# =============================================================================
# Tiny ImageNet-200
# Auto-download: YES  (~235 MB from Stanford CS231n)
# Input: 64x64   Classes: 200   Batch: 64
#
# Non-standard directory structure handled by custom Dataset classes:
#   Train: class/images/*.JPEG  (class-per-folder + extra "images" subfolder)
#   Val  : images/*.JPEG + val_annotations.txt  (flat folder + annotation file)
# =============================================================================

import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from utils.logger import get_logger

logger = get_logger("tiny_imagenet", logfile="logs/tiny_imagenet.log")

_ROOT         = "./data/tiny-imagenet-200"
_DOWNLOAD_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
_FILENAME     = "tiny-imagenet-200.zip"

_MEAN = (0.4802, 0.4481, 0.3975)
_STD  = (0.2764, 0.2689, 0.2816)


def _ensure_downloaded(root: str) -> None:
    """Download and extract Tiny ImageNet if not already present."""
    train_dir = os.path.join(root, "train")
    if os.path.isdir(train_dir):
        return  # already extracted

    logger.info("Tiny ImageNet not found — downloading (~235 MB)...")
    os.makedirs("./data", exist_ok=True)

    try:
        from torchvision.datasets.utils import download_and_extract_archive
        download_and_extract_archive(
            url=_DOWNLOAD_URL,
            download_root="./data",
            extract_root="./data",
            filename=_FILENAME,
            remove_finished=False,
        )
        logger.info("Tiny ImageNet downloaded and extracted to %s", root)
    except Exception as e:
        raise RuntimeError(
            f"Failed to auto-download Tiny ImageNet: {e}\n\n"
            "Manual download:\n"
            "  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
            "  unzip tiny-imagenet-200.zip -d ./data/\n"
        ) from e


# ---------------------------------------------------------------------------
# Custom Dataset wrappers
# ---------------------------------------------------------------------------

class _TinyImageNetTrain(Dataset):
    """
    Reads training images from:
        <root>/train/<class_id>/images/<img>.JPEG
    """
    def __init__(self, root: str, transform=None):
        self.transform   = transform
        self.samples     = []   # (path, class_idx)
        self.class_to_idx = {}

        train_dir  = Path(root) / "train"
        class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            images_dir = class_dir / "images"
            if images_dir.is_dir():
                for img_path in sorted(images_dir.glob("*.JPEG")):
                    self.samples.append((str(img_path), idx))

        logger.info("TinyIN train: %d classes, %d images",
                    len(class_dirs), len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class _TinyImageNetVal(Dataset):
    """
    Reads validation images from:
        <root>/val/images/<filename>.JPEG
    with labels from:
        <root>/val/val_annotations.txt  (tab-separated: filename, class_id, ...)
    """
    def __init__(self, root: str, class_to_idx: dict, transform=None):
        self.transform = transform
        self.samples   = []

        val_dir    = Path(root) / "val"
        images_dir = val_dir / "images"
        ann_file   = val_dir / "val_annotations.txt"

        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                filename, class_id = parts[0], parts[1]
                if class_id not in class_to_idx:
                    continue
                img_path = images_dir / filename
                if img_path.exists():
                    self.samples.append((str(img_path),
                                         class_to_idx[class_id]))

        logger.info("TinyIN val: %d images", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tiny_imagenet_loaders(batch_size=64, num_workers=0, split_test=True,
                               fast_dev_mode=False, root=_ROOT):
    """
    Returns (train_loader, val_loader, test_loader).

    Tiny ImageNet has no labelled test set — test_loader reuses val_loader.
    fast_dev_mode: 9000 train + 500 val images.
    """
    _ensure_downloaded(root)

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

    train_ds = _TinyImageNetTrain(root, transform=transform_train)
    val_ds   = _TinyImageNetVal(root, train_ds.class_to_idx,
                                transform=transform_val)

    if fast_dev_mode:
        np.random.seed(42)
        t_idx = np.random.choice(len(train_ds), size=min(9000, len(train_ds)),
                                 replace=False).tolist()
        v_idx = np.random.choice(len(val_ds),   size=min(500,  len(val_ds)),
                                 replace=False).tolist()
        train_ds = Subset(train_ds, t_idx)
        val_ds   = Subset(val_ds,   v_idx)

    train_loader = _loader(train_ds, shuffle=True)
    val_loader   = _loader(val_ds,   shuffle=False)

    # No test labels → reuse val
    return train_loader, val_loader, val_loader