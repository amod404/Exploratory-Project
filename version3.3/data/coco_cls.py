# data/coco_cls.py
# =============================================================================
# MS COCO 2017 as single-label image classification
# Auto-download: NO  (~20 GB images + pycocotools required)
# Input: 224x224   Classes: 80   Batch: 32
#
# SETUP (one-time):
# ─────────────────
# 1. Install pycocotools:
#      pip install pycocotools          # Linux / Mac
#      pip install pycocotools-windows  # Windows
#
# 2. Download COCO 2017 (~20 GB):
#      mkdir -p ./data/coco/images ./data/coco/annotations
#
#      # Images
#      wget http://images.cocodataset.org/zips/train2017.zip   # ~18 GB
#      wget http://images.cocodataset.org/zips/val2017.zip     # ~1 GB
#      unzip train2017.zip -d ./data/coco/images/
#      unzip val2017.zip   -d ./data/coco/images/
#
#      # Annotations
#      wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
#      unzip annotations_trainval2017.zip -d ./data/coco/
#
#   Final structure:
#      ./data/coco/images/train2017/*.jpg
#      ./data/coco/images/val2017/*.jpg
#      ./data/coco/annotations/instances_train2017.json
#      ./data/coco/annotations/instances_val2017.json
#
# CLASSIFICATION STRATEGY:
# ─────────────────────────
# COCO is a multi-label detection dataset. We convert it to single-label
# classification by assigning each image the category of the annotation
# with the largest bounding-box area.
# =============================================================================

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import numpy as np
from utils.logger import get_logger

logger = get_logger("coco_cls", logfile="logs/coco_cls.log")

_COCO_ROOT  = "./data/coco"
_TRAIN_IMGS = os.path.join(_COCO_ROOT, "images",      "train2017")
_VAL_IMGS   = os.path.join(_COCO_ROOT, "images",      "val2017")
_TRAIN_ANN  = os.path.join(_COCO_ROOT, "annotations", "instances_train2017.json")
_VAL_ANN    = os.path.join(_COCO_ROOT, "annotations", "instances_val2017.json")


def _check_paths():
    missing = []
    for path, label in [
        (_TRAIN_IMGS, "train images"), (_VAL_IMGS, "val images"),
        (_TRAIN_ANN, "train annotations"), (_VAL_ANN, "val annotations"),
    ]:
        if not os.path.exists(path):
            missing.append(f"  MISSING: {path}  ({label})")
    if missing:
        raise FileNotFoundError(
            "COCO dataset not found. Missing files:\n"
            + "\n".join(missing)
            + "\n\nSee setup instructions at the top of data/coco_cls.py"
        )


class COCOClassification(Dataset):
    """Single-label COCO: assigns each image the class of its largest annotation."""

    def __init__(self, img_dir, ann_file, transform=None):
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError(
                "pycocotools is required for COCO.\n"
                "  pip install pycocotools          # Linux / Mac\n"
                "  pip install pycocotools-windows  # Windows"
            )

        self.img_dir   = img_dir
        self.transform = transform
        coco           = COCO(ann_file)

        cat_ids         = sorted(coco.getCatIds())
        self.cat_to_idx = {c: i for i, c in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

        self.samples = []
        for img_id in coco.imgs:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            if not ann_ids:
                continue
            anns  = coco.loadAnns(ann_ids)
            best  = max(anns, key=lambda a: a.get("area", 0))
            label = self.cat_to_idx.get(best["category_id"])
            if label is None:
                continue
            fname    = coco.imgs[img_id]["file_name"]
            img_path = os.path.join(img_dir, fname)
            if os.path.exists(img_path):
                self.samples.append((img_path, label))

        logger.info("COCOClassification: %d images, %d classes",
                    len(self.samples), self.num_classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_coco_loaders(batch_size=32, num_workers=0, split_test=True,
                     fast_dev_mode=False):
    """
    Returns (train_loader, val_loader, test_loader).
    fast_dev_mode: 2000 train + 500 val images.
    """
    _check_paths()   # raises clear error if data is missing

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

    train_ds = COCOClassification(_TRAIN_IMGS, _TRAIN_ANN, transform_train)
    val_ds   = COCOClassification(_VAL_IMGS,   _VAL_ANN,   transform_val)

    if fast_dev_mode:
        np.random.seed(42)
        t_idx = np.random.choice(len(train_ds), size=min(2000, len(train_ds)),
                                 replace=False).tolist()
        v_idx = np.random.choice(len(val_ds),   size=min(500,  len(val_ds)),
                                 replace=False).tolist()
        train_ds = Subset(train_ds, t_idx)
        val_ds   = Subset(val_ds,   v_idx)

    return (
        _loader(train_ds, shuffle=True),
        _loader(val_ds,   shuffle=False),
        _loader(val_ds,   shuffle=False),   # no separate test split
    )