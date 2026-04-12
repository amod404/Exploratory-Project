# main.py
# =============================================================================
# Entry point for LEMONADE NAS.
# ALL hyperparameters live in config.py — edit them there.
#
# Google Colab usage:
#   1. Upload your project folder to Colab (or mount Google Drive)
#   2. Run: !pip install thop tqdm scikit-learn   (one-time setup)
#   3. Run this file as a cell:  !python main.py
#      OR import and call: from main import main; main()
# =============================================================================

import os
import multiprocessing

# ---- Thread-count lock (must happen before any torch import) ----
# On Colab with GPU we don't use subprocesses for training, so these are
# mainly a safety net for CPU fallback workers.
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")

import warnings
warnings.filterwarnings("ignore")

import copy
import pickle
import datetime
import torch

from config import CFG
from evolution.lemonade_full import run_lemonade
from evolution.operators import random_operator
from evolution.individual import Individual
from models.basenet import build_basenet_graph
from utils.logger import get_logger

logger = get_logger("main", logfile="logs/main.log")

# =============================================================================
# Dataset + num_classes wiring
# =============================================================================

_DATASET_META = {
    "CIFAR-10":  (10,   128),
    "CIFAR-100": (100,  128),
    "IMAGENET":  (1000,  32),
}

NUM_CLASSES = _DATASET_META[CFG.TARGET_DATASET][0]
if CFG.TARGET_DATASET == "IMAGENET" and CFG.BATCH_SIZE == 128:
    CFG.BATCH_SIZE = 32   # ImageNet needs smaller batch on GPU

# =============================================================================
# Seed population
# =============================================================================

def create_seed_population(num_seeds: int) -> list:
    logger.info("Building %d seed architectures for %s (%d classes)",
                num_seeds, CFG.TARGET_DATASET, NUM_CLASSES)

    base_graph = build_basenet_graph(
        num_classes=NUM_CLASSES,
        dataset_type=CFG.TARGET_DATASET,
    )
    graphs = [base_graph]

    for _ in range(num_seeds - 1):
        for _ in range(15):
            tmp = Individual(copy.deepcopy(base_graph))
            new_graph, _, _ = random_operator(tmp)
            if new_graph is None:
                continue
            new_ind = Individual(new_graph)
            try:
                cheap = new_ind.evaluate_cheap(
                    objective_keys=CFG.CHEAP_OBJECTIVES,
                    input_size=(1, 3, 32, 32),
                )
                if cheap.get("params", 0) <= CFG.MAX_PARAMS:
                    graphs.append(new_graph)
                    break
            except Exception:
                continue
        else:
            graphs.append(copy.deepcopy(base_graph))

    logger.info("Seed population ready: %d architectures", len(graphs))
    return graphs


# =============================================================================
# Main
# =============================================================================

def main():
    # ------------------------------------------------------------------
    # Device detection with detailed info
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU detected: {gpu_name}  ({gpu_mem:.1f} GB VRAM)")
        print(f"  AMP (fp16): {'ENABLED' if CFG.USE_AMP else 'DISABLED'}")
        print(f"  Training strategy: SEQUENTIAL on GPU (faster than multiprocessing)\n")
    else:
        device = "cpu"
        n_cpu = os.cpu_count() or 1
        print(f"\n  No GPU — using CPU ({n_cpu} cores, parallel training)\n")
        # On CPU, AMP has no effect
        CFG.USE_AMP = False

    logger.info("Starting LEMONADE NAS on %s | device=%s | dataset=%s",
                CFG.TARGET_DATASET, device, CFG.TARGET_DATASET)
    logger.info("Config: %s", CFG)

    # ------------------------------------------------------------------
    # Timestamped output directory
    # ------------------------------------------------------------------
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results_{CFG.TARGET_DATASET}_{ts}"
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # ------------------------------------------------------------------
    # Data loaders for main process
    # ------------------------------------------------------------------
    from data.loader_factory import get_loaders
    loaders     = get_loaders(CFG, split_test=True)
    train_loader, val_loader = loaders[0], loaders[1]
    test_loader = loaders[2] if len(loaders) > 2 else None

    # ------------------------------------------------------------------
    # Seed population
    # ------------------------------------------------------------------
    init_graphs = create_seed_population(CFG.NUM_SEEDS)

    # ------------------------------------------------------------------
    # Run LEMONADE
    # ------------------------------------------------------------------
    final_population, history = run_lemonade(
        init_graphs  = init_graphs,
        cfg          = CFG,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        run_dir      = run_dir,
    )

    # ------------------------------------------------------------------
    # Save history + config
    # ------------------------------------------------------------------
    history_dir = os.path.join(run_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    with open(os.path.join(history_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f)
    with open(os.path.join(run_dir, "config.pkl"), "wb") as f:
        pickle.dump(CFG, f)
    logger.info("History + config saved to %s", run_dir)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    from utils.plot import plot_all_pairs, plot_3d_pareto, plot_convergence
    plot_all_pairs(history, cheap_objectives=CFG.CHEAP_OBJECTIVES, save_dir=plot_dir)
    plot_3d_pareto(history, save_dir=plot_dir)
    plot_convergence(history, save_dir=plot_dir)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  LEMONADE COMPLETE — {CFG.TARGET_DATASET}")
    print(f"  Results → {run_dir}")
    print("=" * 60)

    for i, ind in enumerate(final_population):
        ve = ind.f_exp.get("val_error")  if ind.f_exp   else None
        p  = ind.f_cheap.get("params")   if ind.f_cheap else None
        fl = ind.f_cheap.get("flops")    if ind.f_cheap else None

        te = None
        if test_loader is not None and ind.model is not None:
            from train.evaluate import evaluate_accuracy
            te = evaluate_accuracy(ind.model, test_loader, device=device)

        p_str  = f"{p:>12,}"       if p  is not None else f"{'?':>12}"
        fl_str = f"{int(fl):>14,}" if fl is not None else f"{'?':>14}"
        ve_str = f"{ve:.4f}"       if ve is not None else "?"
        te_str = f"{te:.4f}"       if te is not None else "N/A"
        print(f"  [{i}] params={p_str}  flops={fl_str}  "
              f"val_err={ve_str}  test_err={te_str}")
        logger.info("Final model %d | params=%s flops=%s val_err=%s test_err=%s",
                    i, p_str.strip(), fl_str.strip(), ve_str, te_str)

    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Colab / Linux uses "fork" by default which is fine for CPU workers.
    # "spawn" is required on Windows and is safer if CUDA workers are ever used.
    # We guard with try/except because set_start_method can only be called once.
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # already set — ignore

    main()