# data/loader_factory.py
# =============================================================================
# Single entry-point for data loading.
# Used by BOTH the main process and CPU worker sub-processes.
#
# get_loaders(cfg)            → (train, val, test) for main process
# get_loaders_for_worker(cfg) → (train, val)       for workers (num_workers=0)
# =============================================================================


def get_loaders(cfg, split_test: bool = True):
    """
    Returns (train_loader, val_loader, test_loader) based on cfg.TARGET_DATASET.
    Always returns a 3-tuple; test_loader may equal val_loader for datasets
    without a separate labelled test split.
    """
    ds  = cfg.TARGET_DATASET
    bs  = cfg.BATCH_SIZE
    fdm = cfg.FAST_DEV_MODE

    if ds == "CIFAR-10":
        from data.cifar10 import get_cifar_loaders
        return get_cifar_loaders(
            batch_size=bs, num_workers=0,
            split_test=split_test, fast_dev_mode=fdm)

    elif ds == "CIFAR-100":
        from data.cifar100 import get_cifar100_loaders
        return get_cifar100_loaders(
            batch_size=bs, num_workers=0,
            split_test=split_test, fast_dev_mode=fdm)

    elif ds == "CINIC-10":
        from data.cinic10 import get_cinic10_loaders
        return get_cinic10_loaders(
            batch_size=bs, num_workers=0,
            split_test=split_test, fast_dev_mode=fdm)

    elif ds == "GTSRB":
        from data.gtsrb import get_gtsrb_loaders
        return get_gtsrb_loaders(
            batch_size=bs, num_workers=0,
            split_test=split_test, fast_dev_mode=fdm)

    elif ds == "EUROSAT":
        from data.eurosat import get_eurosat_loaders
        return get_eurosat_loaders(
            batch_size=bs, num_workers=0,
            split_test=split_test, fast_dev_mode=fdm)

    elif ds == "TINY_IMAGENET":
        from data.tiny_imagenet import get_tiny_imagenet_loaders
        return get_tiny_imagenet_loaders(
            batch_size=bs, num_workers=0,
            split_test=split_test, fast_dev_mode=fdm)

    elif ds == "IMAGENET":
        from data.imagenet import get_imagenet_loaders
        return get_imagenet_loaders(
            batch_size=bs, num_workers=0,
            split_test=split_test, fast_dev_mode=fdm)

    elif ds == "PLACES365":
        from data.places365 import get_places365_loaders
        return get_places365_loaders(
            batch_size=bs, num_workers=0,
            split_test=split_test, fast_dev_mode=fdm)

    elif ds == "COCO":
        from data.coco_cls import get_coco_loaders
        return get_coco_loaders(
            batch_size=bs, num_workers=0,
            split_test=split_test, fast_dev_mode=fdm)

    else:
        supported = [
            "CIFAR-10", "CIFAR-100", "CINIC-10", "GTSRB", "EUROSAT",
            "TINY_IMAGENET", "IMAGENET", "PLACES365", "COCO",
        ]
        raise ValueError(
            f"Unknown dataset '{ds}'.\nSupported: {supported}"
        )


def get_loaders_for_worker(cfg):
    """
    Returns (train_loader, val_loader) for use inside a CPU subprocess.
    Always num_workers=0 to prevent nested multiprocessing crashes.
    """
    result = get_loaders(cfg, split_test=True)
    return result[0], result[1]