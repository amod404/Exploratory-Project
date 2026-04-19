# main.py
# =============================================================================
# Entry point for LEMONADE NAS.
# ALL hyperparameters live in config.py — edit them there.
# =============================================================================

import os

# Must be set BEFORE any PyTorch import to prevent thread-count explosion
# in multiprocessing workers.
os.environ["OMP_NUM_THREADS"]         = "1"
os.environ["MKL_NUM_THREADS"]         = "1"
os.environ["OPENBLAS_NUM_THREADS"]    = "1"
os.environ["VECLIB_MAXIMUM_THREADS"]  = "1"
os.environ["NUMEXPR_NUM_THREADS"]     = "1"

import warnings
warnings.filterwarnings("ignore")

import copy
import pickle
import datetime
import torch

from config import CFG                          # ← single source of truth
from evolution.lemonade_full import run_lemonade
from evolution.operators import random_operator
from evolution.individual import Individual
from models.basenet import build_basenet_graph
from utils.logger import get_logger

logger = get_logger("main", logfile="logs/main.log")


# =============================================================================
# Seed population
# =============================================================================

def create_seed_population(num_seeds: int) -> list:
    logger.info(
        "Building %d seed architectures for %s (%d classes, input %s)",
        num_seeds, CFG.TARGET_DATASET, CFG.NUM_CLASSES, CFG.INPUT_SIZE,
    )

    base_graph = build_basenet_graph(
        num_classes=CFG.NUM_CLASSES,
        dataset_type=CFG.TARGET_DATASET,
    )
    graphs = [base_graph]

    for _ in range(num_seeds - 1):
        for _ in range(15):        # up to 15 attempts to produce a valid variant
            tmp = Individual(copy.deepcopy(base_graph))
            new_graph, _, _ = random_operator(tmp)
            if new_graph is None:
                continue
            new_ind = Individual(new_graph)
            try:
                cheap = new_ind.evaluate_cheap(
                    objective_keys=CFG.CHEAP_OBJECTIVES,
                    input_size=CFG.INPUT_SIZE,
                )
                param_count = cheap.get("params", 0)
                if param_count <= CFG.MAX_PARAMS:
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
    logger.info("Starting LEMONADE NAS on %s", CFG.TARGET_DATASET)
    logger.info("Config: %s", CFG)

    # ------------------------------------------------------------------
    # Timestamped output directory
    # ------------------------------------------------------------------
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results_{CFG.TARGET_DATASET}_{ts}"
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    print(f"\n  Dataset  : {CFG.TARGET_DATASET}")
    print(f"  Classes  : {CFG.NUM_CLASSES}")
    print(f"  Input    : {CFG.INPUT_SIZE}")
    print(f"  Batch    : {CFG.BATCH_SIZE}")
    print(f"  Device   : {device}\n")

    # ------------------------------------------------------------------
    # Data loaders  (main process only — workers load their own)
    # ------------------------------------------------------------------
    from data.loader_factory import get_loaders
    loaders = get_loaders(CFG, split_test=True)
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
    history_path = os.path.join(history_dir, "history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    logger.info("History saved → %s", history_path)

    config_path = os.path.join(run_dir, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(CFG, f)

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
    # Final test evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  LEMONADE COMPLETE — {CFG.TARGET_DATASET}")
    print(f"  Results → {run_dir}")
    print("=" * 60)

    for i, ind in enumerate(final_population):
        ve = ind.f_exp.get("val_error") if ind.f_exp else None
        p  = ind.f_cheap.get("params")  if ind.f_cheap else None
        fl = ind.f_cheap.get("flops")   if ind.f_cheap else None

        te = None
        if test_loader is not None and ind.model is not None:
            from train.evaluate import evaluate_accuracy
            te = evaluate_accuracy(ind.model, test_loader, device=device)

        logger.info(
            "Final model %d | id=%s params=%s flops=%s val_err=%s test_err=%s",
            i, ind.id,
            f"{p:,}"        if p  is not None else "?",
            f"{int(fl):,}"  if fl is not None else "?",
            f"{ve:.4f}"     if ve is not None else "?",
            f"{te:.4f}"     if te is not None else "?",
        )
        p_str  = f"{p:>12,}"       if p  is not None else f"{'?':>12}"
        fl_str = f"{int(fl):>14,}" if fl is not None else f"{'?':>14}"
        ve_str = f"{ve:.4f}"       if ve is not None else "?"
        te_str = f"{te:.4f}"       if te is not None else "N/A"
        print(f"  [{i}] params={p_str}  flops={fl_str}  "
              f"val_err={ve_str}  test_err={te_str}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()