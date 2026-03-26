# main.py

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch

from architectures.graph import ArchitectureGraph
from models.base_net import build_sequential_macro_architecture
from evolution.lemonade_full import run_lemonade
from data.cifar10 import get_cifar_loaders
from utils.logger import get_logger

logger = get_logger("main", logfile="logs/main.log")


def seed_graph():
    """
    Generates the initial starting graph using the sequentially 
    stacked cells scheme as outlined for LEMONADE.
    """
    # Start small so the evolutionary morphisms can grow the network
    return build_sequential_macro_architecture(
        in_channels=3,
        init_channels=16,
        num_classes=10,
        num_cells=4, # 4 initial cells stacked sequentially
        image_size=32
    )


def main():

    logger.info("Starting FULL LEMONADE experiment")

    device = "cpu"
    logger.info("Using device: %s", device)

    # -------------------------------------------------
    # Load CIFAR-10
    # -------------------------------------------------
    train_loader, val_loader = get_cifar_loaders(batch_size=128)

    # -------------------------------------------------
    # Run LEMONADE
    # -------------------------------------------------
    final_population = run_lemonade(
        init_graphs=[seed_graph() for _ in range(2)],
        generations=6,
        n_children=10,
        n_accept=6,
        epochs=3,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # -------------------------------------------------
    # Print final results
    # -------------------------------------------------
    logger.info("Final Pareto population size: %d", len(final_population))

    for i, ind in enumerate(final_population):

        val_error = None
        if ind.f_exp is not None:
            val_error = ind.f_exp.get("val_error")

        logger.info(
            "Model %d : params=%d | flops=%d | val_error=%s",
            i,
            ind.f_cheap['params'],
            ind.f_cheap['flops'],
            str(val_error)
        )


if __name__ == "__main__":
    main()