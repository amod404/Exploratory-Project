################################################################################
# FOLDER: Root Directory
# FILE:   main.py
# PATH:   .\main.py
################################################################################

import os
# ==============================================================================
# CRITICAL MULTIPROCESSING FIX: Must be set BEFORE PyTorch is imported.
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore") 

import torch
import copy
from evolution.lemonade_full import run_lemonade
from evolution.operators import random_operator
from utils.logger import get_logger
from evolution.individual import Individual
from utils.plot import plot_all_pairs

from models.basenet import build_basenet_graph

# ==============================================================================
# EASY SWAP CONFIGURATION
# Change this single variable to switch datasets instantly!
# Options: "CIFAR-10", "CIFAR-100", "IMAGENET"
# ==============================================================================
TARGET_DATASET = "IMAGENET"  

if TARGET_DATASET == "IMAGENET":
    from data.imagenet import get_imagenet_loaders as get_loaders
    NUM_CLASSES = 1000
    BATCH_SIZE = 32  # ImageNet images are huge; requires smaller batch sizes to prevent OOM
elif TARGET_DATASET == "CIFAR-100":
    from data.cifar100 import get_cifar100_loaders as get_loaders
    NUM_CLASSES = 100
    BATCH_SIZE = 128
else:
    from data.cifar10 import get_cifar_loaders as get_loaders
    NUM_CLASSES = 10
    BATCH_SIZE = 128
# ==============================================================================

logger = get_logger("main", logfile="logs/main.log")

def create_diverse_seed_population(num_seeds=5):
    logger.info("Generating diverse seed population of size %d for %s (Classes: %d)", num_seeds, TARGET_DATASET, NUM_CLASSES)
    base_graph = build_basenet_graph(num_classes=NUM_CLASSES, dataset_type=TARGET_DATASET)
    population = [base_graph] 
    
    for _ in range(num_seeds - 1):
        for attempt in range(10): 
            temp_ind = Individual(copy.deepcopy(base_graph))
            new_graph, _, _ = random_operator(temp_ind)
            if new_graph is not None:
                new_ind = Individual(new_graph)
                try:
                    cheap_obj = new_ind.evaluate_cheap()
                    if cheap_obj['params'] < 15_000_000: # Slightly relaxed for ImageNet
                        population.append(new_graph)
                        break
                except Exception:
                    continue
        else:
            population.append(copy.deepcopy(base_graph))
            
    return population

def main():
    logger.info("Starting FULL LEMONADE experiment on %s", TARGET_DATASET)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    try:
        train_loader, val_loader, test_loader = get_loaders(
            batch_size=BATCH_SIZE, 
            split_test=True, 
            fast_dev_mode=False  
        )
    except ValueError:
        train_loader, val_loader = get_loaders(batch_size=BATCH_SIZE, fast_dev_mode=False)
        test_loader = None

    init_graphs = create_diverse_seed_population(num_seeds=6)

    final_population, history = run_lemonade(
        init_graphs=init_graphs,
        generations=6,
        n_children=10,   
        n_accept=4,     
        epochs=8,       
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    logger.info("Final Pareto population size: %d", len(final_population))
    plot_all_pairs(history, save_dir="plotLogs")
    
    print("\n" + "="*50)
    print("LEMONADE NAS COMPLETE - FINAL PARETO FRONT")
    print("="*50)

    for i, ind in enumerate(final_population):
        val_error = ind.f_exp.get("val_error") if ind.f_exp else None
        
        test_error = None
        if test_loader is not None:
            from train.evaluate import evaluate_accuracy
            logger.info("Evaluating Final Model %d on Test Set...", i)
            test_error = evaluate_accuracy(ind.build_model(), test_loader, device=device)

        logger.info(
            "Model %d : params=%d | flops=%d | val_error=%s | test_error=%s",
            i, ind.f_cheap['params'], ind.f_cheap['flops'],
            f"{val_error:.4f}" if val_error else "N/A",
            f"{test_error:.4f}" if test_error else "N/A"
        )
        
        print(f"Model {i}: Params: {ind.f_cheap['params']:,} | FLOPs: {ind.f_cheap['flops']:,} | Val Err: {val_error:.4f}")

if __name__ == "__main__":
    main()
# ################################################################################
# # FOLDER: Root Directory
# # FILE:   main.py
# # PATH:   .\main.py
# ################################################################################

# import os
# # ==============================================================================
# # CRITICAL MULTIPROCESSING FIX: Must be set BEFORE PyTorch is imported.
# # Prevents workers from spawning exponential threads and crashing the PC.
# # ==============================================================================
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# import warnings
# warnings.filterwarnings("ignore") 

# import torch
# import copy
# from evolution.lemonade_full import run_lemonade
# from evolution.operators import random_operator
# from data.cifar10 import get_cifar_loaders 
# from utils.logger import get_logger
# from evolution.individual import Individual
# from utils.plot import plot_all_pairs

# # IMPORT YOUR NEW MODULAR GRAPH BUILDER
# from models.basenet import build_basenet_graph

# logger = get_logger("main", logfile="logs/main.log")

# def create_diverse_seed_population(num_seeds=5):
#     logger.info("Generating diverse seed population of size %d", num_seeds)
#     base_graph = build_basenet_graph()
#     population = [base_graph] 
    
#     for _ in range(num_seeds - 1):
#         for attempt in range(10): 
#             temp_ind = Individual(copy.deepcopy(base_graph))
#             new_graph, _, _ = random_operator(temp_ind)
#             if new_graph is not None:
#                 new_ind = Individual(new_graph)
#                 try:
#                     cheap_obj = new_ind.evaluate_cheap()
#                     if cheap_obj['params'] < 10_000_000:
#                         population.append(new_graph)
#                         break
#                 except Exception:
#                     continue
#         else:
#             population.append(copy.deepcopy(base_graph))
            
#     return population

# def main():
#     logger.info("Starting FULL LEMONADE experiment")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.info("Using device: %s", device)

#     try:
#         train_loader, val_loader, test_loader = get_cifar_loaders(
#             batch_size=128, 
#             split_test=True, 
#             fast_dev_mode=True  
#         )
#     except ValueError:
#         train_loader, val_loader = get_cifar_loaders(batch_size=128, fast_dev_mode=True)
#         test_loader = None

#     init_graphs = create_diverse_seed_population(num_seeds=6)

#     final_population, history = run_lemonade(
#         init_graphs=init_graphs,
#         generations=4,
#         n_children=10,   
#         n_accept=5,     
#         epochs=8,       
#         train_loader=train_loader,
#         val_loader=val_loader,
#         device=device,
#     )

#     logger.info("Final Pareto population size: %d", len(final_population))
#     plot_all_pairs(history, save_dir="logs")
    
#     print("\n" + "="*50)
#     print("LEMONADE NAS COMPLETE - FINAL PARETO FRONT")
#     print("="*50)

#     for i, ind in enumerate(final_population):
#         val_error = ind.f_exp.get("val_error") if ind.f_exp else None
        
#         test_error = None
#         if test_loader is not None:
#             from train.evaluate import evaluate_accuracy
#             logger.info("Evaluating Final Model %d on Test Set...", i)
#             test_error = evaluate_accuracy(ind.build_model(), test_loader, device=device)

#         logger.info(
#             "Model %d : params=%d | flops=%d | val_error=%s | test_error=%s",
#             i, ind.f_cheap['params'], ind.f_cheap['flops'],
#             f"{val_error:.4f}" if val_error else "N/A",
#             f"{test_error:.4f}" if test_error else "N/A"
#         )
        
#         print(f"Model {i}: Params: {ind.f_cheap['params']:,} | FLOPs: {ind.f_cheap['flops']:,} | Val Err: {val_error:.4f}")

# if __name__ == "__main__":
#     main()
