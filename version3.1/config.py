# config.py
# =============================================================================
# ALL USER-FACING HYPERPARAMETERS ARE IN THIS ONE FILE.
# Change values here; nothing else needs to be touched.
#
# GPU / Colab notes:
#   - Set USE_AMP=True  for 1.5-2x speedup on any NVIDIA GPU (T4, A100, V100)
#   - Set NUM_DATALOADER_WORKERS=2 on Colab (it has 2 CPU cores for prefetch)
#   - BATCH_SIZE=128 works on Colab T4 for CIFAR; use 64 for TINY-IMAGENET
#   - ProcessPoolExecutor is automatically disabled when CUDA is available;
#     models are trained sequentially on the GPU instead (much faster).
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class NASConfig:

    # -------------------------------------------------------------------------
    # Dataset
    # Options: "CIFAR-10" | "CIFAR-100" | "TINY-IMAGENET" | "IMAGENET"
    #
    #   CIFAR-10       : 32x32,  10 classes,  ~60K images  (auto-download)
    #   CIFAR-100      : 32x32, 100 classes,  ~60K images  (auto-download)
    #   TINY-IMAGENET  : 64x64, 200 classes, ~100K images  (auto-download)
    #   IMAGENET       : 224x224, 1000 classes — must be provided manually
    # -------------------------------------------------------------------------
    TARGET_DATASET: str = "TINY-IMAGENET"

    # True  → small subset of data (fast iteration / debugging)
    # False → full dataset (production runs)
    FAST_DEV_MODE: bool = False

    BATCH_SIZE: int = 128

    # -------------------------------------------------------------------------
    # Cheap Objectives
    # Options: any subset of ["params", "flops"]
    #   ["params", "flops"]  → full multi-objective (needs thop)
    #   ["params"]           → params-only (fastest, recommended for Colab)
    #   []                   → single-objective NAS (val_error only)
    # -------------------------------------------------------------------------
    CHEAP_OBJECTIVES: List[str] = field(default_factory=lambda: ["params","flops"])

    # -------------------------------------------------------------------------
    # Evolution
    # -------------------------------------------------------------------------
    GENERATIONS: int = 10
    NUM_SEEDS: int = 14

    # npc: candidates generated per generation.  Must be > N_ACCEPT.
    N_CHILDREN: int = 30

    # nac: how many pass the KDE filter and enter expensive training.
    # Ratio N_CHILDREN / N_ACCEPT should be at least 3:1.
    N_ACCEPT: int = 10

    # Hard parameter cap; children exceeding this are discarded before training.
    MAX_PARAMS: int = 25_000_000

    # Minimum Pareto population size.
    MIN_POP: int = 4

    # -------------------------------------------------------------------------
    # Training Epochs  (THREE distinct phases)
    #
    #   INIT_EPOCHS   : Gen 0 — training seed population from scratch
    #   CHILD_EPOCHS  : Gen 1+ — standard training for accepted children
    #   DISTILL_EPOCHS: ANM children only — distillation BEFORE CHILD_EPOCHS
    # -------------------------------------------------------------------------
    INIT_EPOCHS: int = 40
    CHILD_EPOCHS: int = 10
    DISTILL_EPOCHS: int = 5

    # Set True to add +1 epoch every 5 generations
    EPOCH_PROGRESSION: bool = True

    # -------------------------------------------------------------------------
    # Optimizer & Regularisation
    # OPTIMIZER options: "sgd" | "adam" | "adamw"
    # -------------------------------------------------------------------------
    OPTIMIZER: str = "sgd"

    INIT_LR: float = 0.01
    CHILD_LR: float = 0.005
    DISTILL_LR: float = 0.001

    WEIGHT_DECAY: float = 1e-4

    DISTILL_TEMPERATURE: float = 3.0
    DISTILL_ALPHA: float = 0.1   # 0.0 = pure KD, 1.0 = pure CE

    # -------------------------------------------------------------------------
    # KDE Sampler bandwidth
    # -------------------------------------------------------------------------
    KDE_BANDWIDTH: float = 0.3

    # -------------------------------------------------------------------------
    # GPU / Performance
    # -------------------------------------------------------------------------
    # Automatic Mixed Precision (fp16 forward + fp32 gradients).
    # Gives 1.5-2x speedup on NVIDIA GPUs with almost no accuracy loss.
    # Automatically disabled if CUDA is not available.
    USE_AMP: bool = True

    # DataLoader worker processes for the MAIN process.
    # Set 2 on Colab (it has 2 prefetch-CPU cores).
    # Set 0 on Windows or if you hit "RuntimeError: DataLoader worker" errors.
    # CPU subprocess workers always use 0 (nested MP is not safe).
    NUM_DATALOADER_WORKERS: int = 2

    # Pin memory for fast CPU→GPU transfers.
    # Automatically forced to False when CUDA is unavailable.
    PIN_MEMORY: bool = True

    # -------------------------------------------------------------------------
    # UX / Debugging
    # -------------------------------------------------------------------------
    SHOW_PROGRESS_BAR: bool = False

    # -------------------------------------------------------------------------
    # Derived helpers — do NOT edit these
    # -------------------------------------------------------------------------

    def get_input_size(self) -> Tuple[int, int, int, int]:
        """
        Returns the (N, C, H, W) dummy input shape for the current dataset.
        Used by CompiledModel shape-inference, FLOPs estimation, and anywhere
        else a concrete tensor shape is needed.
        """
        if self.TARGET_DATASET == "IMAGENET":
            return (1, 3, 224, 224)
        elif self.TARGET_DATASET == "TINY-IMAGENET":
            return (1, 3, 64, 64)
        else:
            # CIFAR-10 and CIFAR-100 both use 32x32
            return (1, 3, 32, 32)

    def get_num_classes(self) -> int:
        """Returns the number of output classes for the current dataset."""
        _map = {
            "CIFAR-10":     10,
            "CIFAR-100":   100,
            "TINY-IMAGENET": 200,
            "IMAGENET":   1000,
        }
        if self.TARGET_DATASET not in _map:
            raise ValueError(
                f"Unknown TARGET_DATASET '{self.TARGET_DATASET}'. "
                f"Valid choices: {list(_map.keys())}"
            )
        return _map[self.TARGET_DATASET]


# Singleton — import this everywhere
CFG = NASConfig()

# # config.py
# # =============================================================================
# # ALL USER-FACING HYPERPARAMETERS ARE IN THIS ONE FILE.
# # Change values here; nothing else needs to be touched.
# #
# # GPU / Colab notes:
# #   - Set USE_AMP=True  for 1.5-2x speedup on any NVIDIA GPU (T4, A100, V100)
# #   - Set NUM_DATALOADER_WORKERS=2 on Colab (it has 2 CPU cores for prefetch)
# #   - BATCH_SIZE=256 works on Colab T4 for CIFAR; use 128 to be safe
# #   - ProcessPoolExecutor is automatically disabled when CUDA is available;
# #     models are trained sequentially on the GPU instead (much faster).
# # =============================================================================

# from dataclasses import dataclass, field
# from typing import List


# @dataclass
# class NASConfig:

#     # -------------------------------------------------------------------------
#     # Dataset
#     # Options: "CIFAR-10" | "CIFAR-100" | "IMAGENET"
#     # -------------------------------------------------------------------------
#     TARGET_DATASET: str = "CIFAR-100"

#     # True  → small subset of data (fast iteration / debugging)
#     # False → full dataset (production runs)
#     FAST_DEV_MODE: bool = False

#     BATCH_SIZE: int = 128

#     # -------------------------------------------------------------------------
#     # Cheap Objectives
#     # Options: any subset of ["params", "flops"]
#     #   ["params", "flops"]  → full multi-objective (needs thop)
#     #   ["params"]           → params-only (fastest, recommended for Colab)
#     #   []                   → single-objective NAS (val_error only)
#     # -------------------------------------------------------------------------
#     CHEAP_OBJECTIVES: List[str] = field(default_factory=lambda: ["params"])

#     # -------------------------------------------------------------------------
#     # Evolution
#     # -------------------------------------------------------------------------
#     GENERATIONS: int = 5
#     NUM_SEEDS: int = 12

#     # npc: candidates generated per generation.  Must be > N_ACCEPT.
#     N_CHILDREN: int = 24

#     # nac: how many pass the KDE filter and enter expensive training.
#     # Ratio N_CHILDREN / N_ACCEPT should be at least 3:1.
#     N_ACCEPT: int = 8

#     # Hard parameter cap; children exceeding this are discarded before training.
#     MAX_PARAMS: int = 10_000_000

#     # Minimum Pareto population size.
#     MIN_POP: int = 3

#     # -------------------------------------------------------------------------
#     # Training Epochs  (THREE distinct phases)
#     #
#     #   INIT_EPOCHS   : Gen 0 — training seed population from scratch
#     #   CHILD_EPOCHS  : Gen 1+ — standard training for accepted children
#     #   DISTILL_EPOCHS: ANM children only — distillation BEFORE CHILD_EPOCHS
#     # -------------------------------------------------------------------------
#     INIT_EPOCHS: int = 20
#     CHILD_EPOCHS: int = 12
#     DISTILL_EPOCHS: int = 5

#     # Set True to add +1 epoch every 5 generations
#     EPOCH_PROGRESSION: bool = True

#     # -------------------------------------------------------------------------
#     # Optimizer & Regularisation
#     # OPTIMIZER options: "sgd" | "adam" | "adamw"
#     # -------------------------------------------------------------------------
#     OPTIMIZER: str = "sgd"

#     INIT_LR: float = 0.01
#     CHILD_LR: float = 0.005
#     DISTILL_LR: float = 0.001

#     WEIGHT_DECAY: float = 1e-4

#     DISTILL_TEMPERATURE: float = 3.0
#     DISTILL_ALPHA: float = 0.0   # 0.0 = pure KD, 1.0 = pure CE

#     # -------------------------------------------------------------------------
#     # KDE Sampler bandwidth
#     # -------------------------------------------------------------------------
#     KDE_BANDWIDTH: float = 0.3

#     # -------------------------------------------------------------------------
#     # GPU / Performance
#     # -------------------------------------------------------------------------
#     # Automatic Mixed Precision (fp16 forward + fp32 gradients).
#     # Gives 1.5-2x speedup on NVIDIA GPUs with almost no accuracy loss.
#     # Automatically disabled if CUDA is not available.
#     USE_AMP: bool = True

#     # DataLoader worker processes for the MAIN process.
#     # Set 2 on Colab (it has 2 prefetch-CPU cores).
#     # Set 0 on Windows or if you hit "RuntimeError: DataLoader worker" errors.
#     # CPU subprocess workers always use 0 (nested MP is not safe).
#     NUM_DATALOADER_WORKERS: int = 2

#     # Pin memory for fast CPU→GPU transfers.
#     # Automatically forced to False when CUDA is unavailable.
#     PIN_MEMORY: bool = True

#     # -------------------------------------------------------------------------
#     # UX / Debugging
#     # -------------------------------------------------------------------------
#     SHOW_PROGRESS_BAR: bool = True


# # Singleton — import this everywhere
# CFG = NASConfig()