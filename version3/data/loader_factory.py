# data/loader_factory.py
# =============================================================================
# Single entry-point for data loading, used by BOTH the main process and
# worker sub-processes.  Fixes the bug where workers always loaded CIFAR-10
# regardless of TARGET_DATASET.
# =============================================================================


def get_loaders(cfg, split_test: bool = True):
    """Return (train_loader, val_loader[, test_loader]) based on cfg."""
    ds = cfg.TARGET_DATASET
    bs = cfg.BATCH_SIZE
    fdm = cfg.FAST_DEV_MODE

    if ds == "CIFAR-100":
        from data.cifar100 import get_cifar100_loaders
        return get_cifar100_loaders(batch_size=bs, num_workers=0,
                                    split_test=split_test, fast_dev_mode=fdm)
    elif ds == "IMAGENET":
        from data.imagenet import get_imagenet_loaders
        return get_imagenet_loaders(batch_size=bs, num_workers=0,
                                    split_test=split_test, fast_dev_mode=fdm)
    else:  # default: CIFAR-10
        from data.cifar10 import get_cifar_loaders
        return get_cifar_loaders(batch_size=bs, num_workers=0,
                                 split_test=split_test, fast_dev_mode=fdm)


def get_loaders_for_worker(cfg):
    """
    Convenience wrapper that returns only (train_loader, val_loader).
    Workers always use num_workers=0 to avoid nested multiprocessing.
    """
    result = get_loaders(cfg, split_test=True)
    # result is (train, val, test) or (train, val)
    return result[0], result[1]