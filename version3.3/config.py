# config.py
# =============================================================================
# ALL USER-FACING HYPERPARAMETERS ARE IN THIS ONE FILE.
#
# Supported datasets (all auto-download unless noted):
#
#   CIFAR-10      32x32   10 cls  ~170 MB  auto-download
#   CIFAR-100     32x32  100 cls  ~170 MB  auto-download
#   CINIC-10      32x32   10 cls  ~426 MB  auto-download  (CIFAR-10 + ImageNet mix)
#   GTSRB         32x32   43 cls  ~300 MB  auto-download  (traffic signs)
#   EUROSAT       64x64   10 cls   ~90 MB  auto-download  (satellite imagery)
#   TINY_IMAGENET 64x64  200 cls  ~235 MB  auto-download
#   IMAGENET     224x224 1000 cls  150 GB  manual download (FakeData fallback if absent)
#   PLACES365    224x224  365 cls   24 GB  auto-download (small 256px version)
#   COCO         224x224   80 cls   20 GB  manual download + pycocotools
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Per-dataset lookup tables  (auto-populated into __post_init__)
# ---------------------------------------------------------------------------

_INPUT_SIZES = {
    "CIFAR-10":      (1, 3, 32,  32),
    "CIFAR-100":     (1, 3, 32,  32),
    "CINIC-10":      (1, 3, 32,  32),
    "GTSRB":         (1, 3, 32,  32),
    "EUROSAT":       (1, 3, 64,  64),
    "TINY_IMAGENET": (1, 3, 64,  64),
    "IMAGENET":      (1, 3, 224, 224),
    "PLACES365":     (1, 3, 224, 224),
    "COCO":          (1, 3, 224, 224),
}

_NUM_CLASSES = {
    "CIFAR-10":      10,
    "CIFAR-100":    100,
    "CINIC-10":      10,
    "GTSRB":         43,
    "EUROSAT":       10,
    "TINY_IMAGENET": 200,
    "IMAGENET":     1000,
    "PLACES365":     365,
    "COCO":           80,
}

# Optimal batch sizes for NAS (balance speed vs. memory on a typical 8 GB GPU)
_DEFAULT_BATCH = {
    "CIFAR-10":      128,
    "CIFAR-100":     128,
    "CINIC-10":      128,
    "GTSRB":          32,
    "EUROSAT":        64,
    "TINY_IMAGENET":  64,
    "IMAGENET":       32,
    "PLACES365":      32,
    "COCO":           32,
}


@dataclass
class NASConfig:

    # -------------------------------------------------------------------------
    # Dataset — change this one field to switch datasets
    # -------------------------------------------------------------------------
    TARGET_DATASET: str = "EUROSAT"

    # True  → small subset (fast iteration / debugging)
    # False → full dataset (production)
    FAST_DEV_MODE: bool = False

    # Set to 0 to use the dataset-specific default in _DEFAULT_BATCH.
    BATCH_SIZE: int = 0

    # -------------------------------------------------------------------------
    # Cheap Objectives
    # Options: any subset of ["params", "flops"]
    #   ["params", "flops"]  → full multi-objective (needs thop installed)
    #   ["params"]           → params-only (fastest, recommended for CPUs)
    #   []                   → single-objective NAS (val_error only)
    # -------------------------------------------------------------------------
    CHEAP_OBJECTIVES: List[str] = field(default_factory=lambda: ["params","flops"])

    # -------------------------------------------------------------------------
    # Evolution
    # -------------------------------------------------------------------------
    GENERATIONS: int = 15
    NUM_SEEDS: int = 7

    # npc: candidates generated each generation  (must be > N_ACCEPT)
    N_CHILDREN: int = 21

    # nac: candidates that pass KDE filter → enter expensive training
    # Ratio N_CHILDREN / N_ACCEPT should be ≥ 3:1 for meaningful filtering
    N_ACCEPT: int = 7

    # Models with more params than this are discarded before training
    MAX_PARAMS: int = 10_000_000

    # Minimum Pareto population size (diversity fill keeps it at least this large)
    MIN_POP: int = 3

    # -------------------------------------------------------------------------
    # Training Epochs  (THREE separate phases — do NOT mix them)
    #
    #   INIT_EPOCHS   : Gen 0 — seed population trained from scratch
    #   CHILD_EPOCHS  : Gen 1+ — every accepted child trained
    #   DISTILL_EPOCHS: ANM children only — distillation BEFORE CHILD_EPOCHS
    # -------------------------------------------------------------------------
    INIT_EPOCHS: int = 1
    CHILD_EPOCHS: int = 1
    DISTILL_EPOCHS: int = 1

    # Add +1 epoch every 5 generations for slow convergence boost
    EPOCH_PROGRESSION: bool = True

    # -------------------------------------------------------------------------
    # Optimizer  (options: "sgd" | "adam" | "adamw")
    # -------------------------------------------------------------------------
    OPTIMIZER: str = "sgd"

    INIT_LR: float   = 0.01     # Gen 0 seed training
    CHILD_LR: float  = 0.005    # Gen 1+ child training
    DISTILL_LR: float = 0.001   # Distillation alignment

    WEIGHT_DECAY: float = 1e-4

    DISTILL_TEMPERATURE: float = 3.0
    DISTILL_ALPHA: float = 0.1  # 0.0 = pure teacher-mimicking, 1.0 = pure CE

    # -------------------------------------------------------------------------
    # KDE Sampler bandwidth
    # -------------------------------------------------------------------------
    KDE_BANDWIDTH: float = 0.3

    # -------------------------------------------------------------------------
    # UX / Debugging
    # -------------------------------------------------------------------------
    # Controls ALL progress bars: both the outer model-level bar in the main
    # process AND the batch-level bars inside training loops.
    SHOW_PROGRESS_BAR: bool = False

    # -------------------------------------------------------------------------
    # Derived fields — auto-set in __post_init__, do NOT edit manually
    # -------------------------------------------------------------------------
    INPUT_SIZE: Tuple = field(init=False)
    NUM_CLASSES: int  = field(init=False)

    def __post_init__(self):
        if self.TARGET_DATASET not in _INPUT_SIZES:
            supported = sorted(_INPUT_SIZES.keys())
            raise ValueError(
                f"Unknown dataset '{self.TARGET_DATASET}'.\n"
                f"Supported datasets: {supported}"
            )
        self.INPUT_SIZE  = _INPUT_SIZES[self.TARGET_DATASET]
        self.NUM_CLASSES = _NUM_CLASSES[self.TARGET_DATASET]
        if self.BATCH_SIZE == 0:
            self.BATCH_SIZE = _DEFAULT_BATCH[self.TARGET_DATASET]


# Singleton — import this everywhere
CFG = NASConfig()