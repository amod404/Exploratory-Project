# data/tiny_imagenet.py
# =============================================================================
# Tiny ImageNet data loader.
#
#   - 200 classes, 64×64 RGB images
#   - 100,000 train images  (500 per class)
#   - 10,000  val   images  (50  per class)
#
# The dataset is automatically downloaded and prepared the first time it
# is requested.  The val folder is reorganised from the flat layout that
# ships in the zip into the ImageFolder-compatible class-subfolder layout
# so that torchvision.datasets.ImageFolder works out of the box.
#
# Download source: http://cs231n.stanford.edu/tiny-imagenet-200.zip (~237 MB)
# =============================================================================

import os
import shutil
import zipfile
import urllib.request
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

_TINY_URL   = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
_ZIP_NAME   = "tiny-imagenet-200.zip"
_DATA_DIR   = "./data/tiny-imagenet-200"


# ---------------------------------------------------------------------------
# Download + preparation
# ---------------------------------------------------------------------------

def _show_progress(block_count, block_size, total_size):
    """Simple progress callback for urllib.request.urlretrieve."""
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1e6
        tot = total_size  / 1e6
        print(f"\r  Downloading Tiny ImageNet: {mb:.1f}/{tot:.1f} MB "
              f"({pct:.1f}%)    ", end="", flush=True)


def _reorganise_val(val_dir: str):
    """
    The shipped val folder is flat:
        val/images/val_0.JPEG  ...  val_9999.JPEG
        val/val_annotations.txt

    We move images into class subfolders so ImageFolder works:
        val/<wnid>/val_N.JPEG
    """
    images_dir   = os.path.join(val_dir, "images")
    ann_file     = os.path.join(val_dir, "val_annotations.txt")
    sentinel     = os.path.join(val_dir, "_reorganised")

    if os.path.exists(sentinel):
        return  # already done

    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"Expected val annotation file at {ann_file}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Expected val images dir at {images_dir}")

    print("  Reorganising Tiny ImageNet val folder (one-time) …", flush=True)

    # Parse annotation file: filename  wnid  x1 y1 x2 y2
    img_to_class = {}
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_to_class[parts[0]] = parts[1]

    # Create class subdirectories and move images
    for img_name, wnid in img_to_class.items():
        class_dir = os.path.join(val_dir, wnid)
        os.makedirs(class_dir, exist_ok=True)
        src = os.path.join(images_dir, img_name)
        dst = os.path.join(class_dir,  img_name)
        if os.path.exists(src):
            shutil.move(src, dst)

    # Remove now-empty images directory
    try:
        os.rmdir(images_dir)
    except OSError:
        pass

    # Write sentinel so we don't redo this on every run
    open(sentinel, "w").close()
    print("  Val folder reorganised.", flush=True)


def _download_and_prepare(root: str = "./data"):
    """Download Tiny ImageNet zip if needed, extract, and fix val layout."""
    zip_path  = os.path.join(root, _ZIP_NAME)
    data_path = os.path.join(root, "tiny-imagenet-200")
    train_dir = os.path.join(data_path, "train")
    val_dir   = os.path.join(data_path, "val")

    # ── Step 1: Download ──────────────────────────────────────────────
    if not os.path.isdir(train_dir):
        os.makedirs(root, exist_ok=True)

        if not os.path.isfile(zip_path):
            print(f"  Tiny ImageNet not found. Downloading from:\n  {_TINY_URL}")
            print("  (~237 MB — this may take a few minutes on Colab)")
            try:
                urllib.request.urlretrieve(_TINY_URL, zip_path, _show_progress)
                print()  # newline after progress bar
            except Exception as e:
                # Clean up partial download
                try:
                    os.remove(zip_path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"Failed to download Tiny ImageNet: {e}\n"
                    "Please download manually from:\n"
                    f"  {_TINY_URL}\n"
                    f"and place the zip at: {zip_path}"
                ) from e
        else:
            print(f"  Found existing zip at {zip_path}, extracting …")

        # ── Step 2: Extract ───────────────────────────────────────────
        print(f"  Extracting to {root} …", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        print("  Extraction complete.", flush=True)

        # Remove zip to save disk space
        try:
            os.remove(zip_path)
        except OSError:
            pass

    # ── Step 3: Reorganise val ────────────────────────────────────────
    _reorganise_val(val_dir)


# ---------------------------------------------------------------------------
# Public loader function
# ---------------------------------------------------------------------------

def get_tiny_imagenet_loaders(batch_size=128, num_workers=2, pin_memory=False,
                               split_test=False, fast_dev_mode=False):
    """
    Returns Tiny ImageNet data loaders.

    Parameters
    ----------
    batch_size    : mini-batch size
    num_workers   : prefetch workers (use 2 on Colab GPU, 0 on CPU-only)
    pin_memory    : pin CPU tensors for fast GPU transfers
    split_test    : if True, returns (train, val, test); else (train, val)
    fast_dev_mode : if True, uses ~2% of data for quick iteration
    """
    _download_and_prepare(root="./data")

    train_dir = os.path.join(_DATA_DIR, "train")
    val_dir   = os.path.join(_DATA_DIR, "val")

    # ── Transforms ────────────────────────────────────────────────────
    # ImageNet-style normalisation works well for Tiny ImageNet
    mean = (0.4802, 0.4481, 0.3975)
    std  = (0.2770, 0.2691, 0.2821)

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── Dataset objects ───────────────────────────────────────────────
    train_data_full = torchvision.datasets.ImageFolder(
        root=train_dir, transform=transform_train
    )
    val_data_full = torchvision.datasets.ImageFolder(
        root=val_dir, transform=transform_test
    )

    persistent = (num_workers > 0)

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            drop_last=False,
        )

    # ── Fast-dev mode: tiny subsets ───────────────────────────────────
    if fast_dev_mode:
        n_train = max(1, int(0.02 * len(train_data_full)))  # ~2000 samples
        n_val   = max(1, int(0.05 * len(val_data_full)))    # ~500  samples
        np.random.seed(42)
        train_idx = np.random.choice(len(train_data_full), n_train, replace=False).tolist()
        val_idx   = np.random.choice(len(val_data_full),   n_val,   replace=False).tolist()
        train_data_full = Subset(train_data_full, train_idx)
        val_data_full   = Subset(val_data_full,   val_idx)

    if split_test:
        # Use the official val split as test; carve out 10% of train as val
        full_train = train_data_full
        n          = len(full_train)
        np.random.seed(42)
        indices    = np.random.permutation(n).tolist()

        if fast_dev_mode:
            split = max(1, int(0.8 * n))
        else:
            split = int(0.9 * n)

        train_idx = indices[:split]
        val_idx   = indices[split:]

        # We need the base dataset (not a Subset) for the val transform
        if isinstance(full_train, Subset):
            base_ds   = full_train.dataset
            base_idxs = full_train.indices
            val_base  = torchvision.datasets.ImageFolder(
                root=train_dir, transform=transform_test
            )
            actual_train_idx = [base_idxs[i] for i in train_idx]
            actual_val_idx   = [base_idxs[i] for i in val_idx]
            train_set = Subset(full_train.dataset, actual_train_idx)
            val_set   = Subset(val_base,            actual_val_idx)
        else:
            val_base = torchvision.datasets.ImageFolder(
                root=train_dir, transform=transform_test
            )
            train_set = Subset(full_train, train_idx)
            val_set   = Subset(val_base,   val_idx)

        train_loader = make_loader(train_set,   shuffle=True)
        val_loader   = make_loader(val_set,     shuffle=False)
        test_loader  = make_loader(val_data_full, shuffle=False)
        return train_loader, val_loader, test_loader

    else:
        train_loader = make_loader(train_data_full, shuffle=True)
        val_loader   = make_loader(val_data_full,   shuffle=False)
        return train_loader, val_loader