# config.py
# =============================================================================
# ALL USER-FACING HYPERPARAMETERS ARE IN THIS ONE FILE.
# Change values here; nothing else needs to be touched.
#
# GPU / Colab notes:
#   - Set USE_AMP=True  for 1.5-2x speedup on any NVIDIA GPU (T4, A100, V100)
#   - Set NUM_DATALOADER_WORKERS=2 on Colab (it has 2 CPU cores for prefetch)
#   - BATCH_SIZE=256 works on Colab T4 for CIFAR; use 128 to be safe
#   - ProcessPoolExecutor is automatically disabled when CUDA is available;
#     models are trained sequentially on the GPU instead (much faster).
# =============================================================================

from dataclasses import dataclass, field
from typing import List


@dataclass
class NASConfig:

    # -------------------------------------------------------------------------
    # Dataset
    # Options: "CIFAR-10" | "CIFAR-100" | "IMAGENET"
    # -------------------------------------------------------------------------
    TARGET_DATASET: str = "CIFAR-100"

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
    CHEAP_OBJECTIVES: List[str] = field(default_factory=lambda: ["params"])

    # -------------------------------------------------------------------------
    # Evolution
    # -------------------------------------------------------------------------
    GENERATIONS: int = 5
    NUM_SEEDS: int = 12

    # npc: candidates generated per generation.  Must be > N_ACCEPT.
    N_CHILDREN: int = 24

    # nac: how many pass the KDE filter and enter expensive training.
    # Ratio N_CHILDREN / N_ACCEPT should be at least 3:1.
    N_ACCEPT: int = 8

    # Hard parameter cap; children exceeding this are discarded before training.
    MAX_PARAMS: int = 10_000_000

    # Minimum Pareto population size.
    MIN_POP: int = 3

    # -------------------------------------------------------------------------
    # Training Epochs  (THREE distinct phases)
    #
    #   INIT_EPOCHS   : Gen 0 — training seed population from scratch
    #   CHILD_EPOCHS  : Gen 1+ — standard training for accepted children
    #   DISTILL_EPOCHS: ANM children only — distillation BEFORE CHILD_EPOCHS
    # -------------------------------------------------------------------------
    INIT_EPOCHS: int = 20
    CHILD_EPOCHS: int = 12
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
    DISTILL_ALPHA: float = 0.0   # 0.0 = pure KD, 1.0 = pure CE

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
    SHOW_PROGRESS_BAR: bool = True


# Singleton — import this everywhere
CFG = NASConfig()