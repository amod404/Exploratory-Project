# evolution/lemonade_full.py
# =============================================================================
# CORRECT LEMONADE LOOP ORDER (per Algorithm 1 in the paper):
#
#   Gen 0:
#     1. Train seed population (INIT_EPOCHS) in parallel
#     2. Compute initial Pareto front
#
#   Gen 1 .. GENERATIONS:
#     1. KDE fit on population's CHEAP objectives
#     2. Sample npc PARENTS  (inverse density → fill sparse regions)
#     3. Apply ONE operator per parent → npc candidate CHILDREN (no training)
#     4. Evaluate CHEAP objectives on all npc candidates  (free)
#     5. KDE filter: sample nac accepted children from candidates (inverse density)
#        → only nac children enter the expensive training phase
#     6. TRAIN accepted children (in parallel):
#           ANM children: distill(DISTILL_EPOCHS) → train(CHILD_EPOCHS)
#           NM  children: train(CHILD_EPOCHS) only
#     7. Merge into population → compute new Pareto front
#     8. Fill to MIN_POP using diversity-maximising selection if needed
#     9. Save history + model weights
# =============================================================================

import os
import gc
import pickle
import time
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from evolution.individual import Individual
from evolution.pareto import pareto_front, fill_with_diversity
from evolution.sampling import KDESampler
from evolution.operators import random_operator, is_approx_op
from utils.logger import get_logger

logger       = get_logger("lemonade",        logfile="logs/lemonade.log")
error_logger = get_logger("lemonade_errors", logfile="logs/lemonade_errors.log")


# =============================================================================
# Worker function  (runs in a separate subprocess)
# =============================================================================

def _worker_train_child(idx, pickled_payload, cfg, train_epochs: int,
                        train_lr: float):
    """
    Complete child lifecycle inside one worker process:
      load graph → build model ONCE → load inherited weights →
      [distill if ANM] → standard training → evaluate → return.

    Notes
    -----
    * The model is built exactly ONCE.  Distillation and training operate on
      the SAME nn.Module object so inherited weights are never discarded.
    * child.model is cleared before pickling (nn.Module is not safely
      picklable); the trained state dict is written to a temp file instead.
    * No artificial sleep().  No hardcoded dataset.
    """
    import os, gc, pickle, time, traceback
    import torch

    # Silence tqdm in sub-processes unconditionally — avoid pipe crashes
    os.environ["TQDM_DISABLE"] = "1"
    torch.set_num_threads(1)

    try:
        start = time.time()

        child_graph, child_sd_path, parent_graph, parent_sd_path, is_approx = \
            pickle.loads(pickled_payload)

        from objectives.cheap import clean_state_dict

        # ------------------------------------------------------------------
        # 1. Build child model ONCE
        # ------------------------------------------------------------------
        from evolution.individual import Individual as _Ind
        child       = _Ind(child_graph)
        child_model = child.build_model()      # child.model is now set

        # ------------------------------------------------------------------
        # 2. Load Lamarckian inherited weights
        # ------------------------------------------------------------------
        if child_sd_path and os.path.exists(child_sd_path):
            try:
                sd = torch.load(child_sd_path, map_location="cpu",
                                weights_only=True)
                child_model.load_state_dict(clean_state_dict(sd), strict=False)
            except Exception as e:
                pass   # proceed with random init — better than crashing
            finally:
                try:
                    os.remove(child_sd_path)
                except OSError:
                    pass

        # ------------------------------------------------------------------
        # 3. Build teacher model if ANM
        # ------------------------------------------------------------------
        teacher_model = None
        if is_approx and parent_graph is not None:
            try:
                teacher = _Ind(parent_graph)
                teacher_model = teacher.build_model()
                if parent_sd_path and os.path.exists(parent_sd_path):
                    sd = torch.load(parent_sd_path, map_location="cpu",
                                    weights_only=True)
                    teacher_model.load_state_dict(clean_state_dict(sd),
                                                  strict=False)
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False
            except Exception as e:
                teacher_model = None  # distillation is optional; don't crash

        # ------------------------------------------------------------------
        # 4. Evaluate cheap objectives  (fast — uses existing child.model)
        # ------------------------------------------------------------------
        try:
            child.evaluate_cheap(
                objective_keys=cfg.CHEAP_OBJECTIVES,
                input_size=(1, 3, 32, 32),
            )
        except Exception as e:
            child.f_cheap = {k: 0 for k in cfg.CHEAP_OBJECTIVES}

        # ------------------------------------------------------------------
        # 5. Load dataset (correct dataset based on cfg)
        # ------------------------------------------------------------------
        from data.loader_factory import get_loaders_for_worker
        train_loader_w, val_loader_w = get_loaders_for_worker(cfg)

        device = "cpu"  # workers always run on CPU (avoids CUDA fork issues)

        # ------------------------------------------------------------------
        # 6a. Distillation phase  (ANM only, DISTILL_EPOCHS, DISTILL_LR)
        #     Operates on child_model directly — NO new model build
        # ------------------------------------------------------------------
        if is_approx and teacher_model is not None:
            try:
                from train.distill import train_with_distillation
                train_with_distillation(
                    student_model=child_model,
                    teacher_model=teacher_model,
                    train_loader=train_loader_w,
                    device=device,
                    epochs=cfg.DISTILL_EPOCHS,
                    lr=cfg.DISTILL_LR,
                    temperature=cfg.DISTILL_TEMPERATURE,
                    alpha=cfg.DISTILL_ALPHA,
                    weight_decay=cfg.WEIGHT_DECAY,
                    optimizer_name=cfg.OPTIMIZER,
                    show_progress=False,        # always off in workers
                )
            except Exception as e:
                error_logger.error("Distillation failed for %s: %s",
                                   child.id, e)
            finally:
                del teacher_model
                gc.collect()

        # ------------------------------------------------------------------
        # 6b. Standard training phase  (all children, train_epochs, train_lr)
        #     Operates on child_model directly
        # ------------------------------------------------------------------
        try:
            from train.trainer import train_model
            train_model(
                model=child_model,
                train_loader=train_loader_w,
                device=device,
                epochs=train_epochs,
                lr=train_lr,
                weight_decay=cfg.WEIGHT_DECAY,
                optimizer_name=cfg.OPTIMIZER,
                show_progress=False,
            )
        except Exception as e:
            error_logger.error("Training failed for %s: %s", child.id, e)

        # ------------------------------------------------------------------
        # 7. Evaluate accuracy  (uses child_model — same object)
        # ------------------------------------------------------------------
        try:
            from train.evaluate import evaluate_accuracy
            val_error = evaluate_accuracy(child_model, val_loader_w,
                                          device=device)
        except Exception as e:
            val_error = 1.0   # worst case; model still enters Pareto candidate pool

        child.f_exp = {"val_error": val_error}

        # ------------------------------------------------------------------
        # 8. Persist trained weights to disk
        # ------------------------------------------------------------------
        trained_sd_path = os.path.join(
            tempfile.gettempdir(),
            f"nas_trained_{child.id}_{int(time.time() * 1000)}.pt",
        )
        torch.save(child_model.state_dict(), trained_sd_path)

        # ------------------------------------------------------------------
        # 9. Clear model before pickling (nn.Module is not picklable)
        # ------------------------------------------------------------------
        child.model = None
        del child_model
        gc.collect()

        return {
            "idx":             idx,
            "status":          "ok",
            "pickled_child":   pickle.dumps(child),
            "trained_sd_path": trained_sd_path,
            "duration":        time.time() - start,
        }

    except Exception as exc:
        return {
            "idx":       idx,
            "status":    "error",
            "error":     str(exc),
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# Helpers
# =============================================================================

def _parallel_train(payloads, cfg, train_epochs, train_lr, desc):
    """
    Submit payloads to a process pool and collect trained Individuals.
    Returns list of Individual objects with model weights loaded.
    """
    import torch
    cpu_count  = max(1, (os.cpu_count() or 2) - 1)
    max_workers = min(cpu_count, len(payloads))

    trained = []

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(_worker_train_child, idx, pc, cfg,
                       train_epochs, train_lr): idx
            for idx, pc in payloads
        }

        if cfg.SHOW_PROGRESS_BAR:
            try:
                from tqdm import tqdm
                iterator = tqdm(as_completed(futures), total=len(futures),
                                desc=desc, unit="model")
            except ImportError:
                iterator = as_completed(futures)
        else:
            iterator = as_completed(futures)

        for fut in iterator:
            try:
                result = fut.result()
            except Exception as e:
                error_logger.error("%s future exception: %s", desc, e)
                continue

            if result.get("status") != "ok":
                error_logger.error("%s worker error [idx=%s]: %s\n%s",
                                   desc,
                                   result.get("idx"),
                                   result.get("error"),
                                   result.get("traceback", ""))
                continue

            try:
                ind = pickle.loads(result["pickled_child"])
                sd_path = result.get("trained_sd_path")
                if sd_path and os.path.exists(sd_path):
                    # Build model fresh in main process then load trained weights
                    model = ind.build_model()
                    model.load_state_dict(
                        torch.load(sd_path, map_location="cpu",
                                   weights_only=True),
                        strict=False,
                    )
                    try:
                        os.remove(sd_path)
                    except OSError:
                        pass
                trained.append(ind)
            except Exception as e:
                error_logger.error("Failed to deserialise trained child: %s", e)

    return trained


def _save_generation_models(population, gen, models_dir):
    """
    Save model weights and graph metadata for every individual in population.
    Returns a list of dicts (the history entry for this generation).
    """
    import torch
    gen_dir = os.path.join(models_dir, f"gen_{gen:03d}")
    os.makedirs(gen_dir, exist_ok=True)

    history_entry = []
    for ind in population:
        record = {
            "id":        ind.id,
            "params":    ind.f_cheap.get("params") if ind.f_cheap else None,
            "flops":     ind.f_cheap.get("flops")  if ind.f_cheap else None,
            "val_error": ind.f_exp.get("val_error") if ind.f_exp   else None,
            "model_path": None,
            "graph_path": None,
        }
        # Save weights if model is built
        if ind.model is not None:
            try:
                wpath = os.path.join(gen_dir, f"{ind.id}_weights.pt")
                torch.save(ind.model.state_dict(), wpath)
                record["model_path"] = wpath
            except Exception:
                pass
        # Save graph (lightweight pickle)
        try:
            gpath = os.path.join(gen_dir, f"{ind.id}_graph.pkl")
            with open(gpath, "wb") as f:
                pickle.dump(ind.graph, f)
            record["graph_path"] = gpath
        except Exception:
            pass

        history_entry.append(record)

    return history_entry


def _print_summary(gen, population):
    rows = []
    for ind in population:
        p = ind.f_cheap.get("params") if ind.f_cheap else None
        f = ind.f_cheap.get("flops")  if ind.f_cheap else None
        v = ind.f_exp.get("val_error") if ind.f_exp   else None
        rows.append((p, f, v))
    rows.sort(key=lambda r: (r[2] if r[2] is not None else 1.0,
                             r[0] if r[0] is not None else float("inf")))
    print(f"\n{'='*65}")
    print(f"  Generation {gen}  |  Pareto population: {len(rows)} models")
    print(f"  {'params':>10}  {'flops':>12}  {'val_error':>10}")
    for p, f, v in rows[:8]:
        ps = f"{p:>10,}" if p is not None else f"{'?':>10}"
        fs = f"{int(f):>12,}" if f is not None else f"{'?':>12}"
        vs = f"{v:.4f}" if v is not None else "?"
        print(f"  {ps}  {fs}  {vs:>10}")
    print(f"{'='*65}\n")


# =============================================================================
# Main LEMONADE entry point
# =============================================================================

def run_lemonade(init_graphs, cfg, train_loader, val_loader, device, run_dir):
    """
    Run the LEMONADE algorithm.

    Parameters
    ----------
    init_graphs  : list[ArchitectureGraph]  — seed architectures
    cfg          : NASConfig
    train_loader : DataLoader (main process only; workers load their own)
    val_loader   : DataLoader (main process only)
    device       : "cpu" or "cuda"
    run_dir      : str — root output directory for this experiment

    Returns
    -------
    (final_population: list[Individual], history: dict)
    """
    import torch

    logger.info(
        "LEMONADE start: gens=%d N_pc=%d N_ac=%d "
        "init_ep=%d child_ep=%d distill_ep=%d device=%s",
        cfg.GENERATIONS, cfg.N_CHILDREN, cfg.N_ACCEPT,
        cfg.INIT_EPOCHS, cfg.CHILD_EPOCHS, cfg.DISTILL_EPOCHS, device,
    )

    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    temp_dir   = tempfile.gettempdir()
    history    = {}
    sampler    = KDESampler(base_bandwidth=cfg.KDE_BANDWIDTH)

    # ------------------------------------------------------------------
    # Initialise population
    # ------------------------------------------------------------------
    population = [Individual(g) for g in init_graphs]

    for ind in population:
        try:
            ind.evaluate_cheap(
                objective_keys=cfg.CHEAP_OBJECTIVES,
                input_size=(1, 3, 32, 32),
            )
        except Exception as e:
            ind.f_cheap = {k: 0 for k in cfg.CHEAP_OBJECTIVES}
            error_logger.error("Cheap eval failed for seed %s: %s", ind.id, e)

    # ------------------------------------------------------------------
    # Generation 0 — train seed population
    # ------------------------------------------------------------------
    logger.info("=== Generation 0: Training %d seed models ===", len(population))
    payloads_g0 = []
    for idx, ind in enumerate(population):
        try:
            if ind.model is None:
                ind.build_model()
            sd_path = os.path.join(temp_dir, f"seed_{ind.id}.pt")
            torch.save(ind.model.state_dict(), sd_path)
            # payload: (child_graph, child_sd_path, parent_graph, parent_sd_path, is_approx)
            pc = pickle.dumps((ind.graph, sd_path, None, None, False))
            payloads_g0.append((idx, pc))
        except Exception as e:
            error_logger.error("Gen 0 serialise error for %s: %s", ind.id, e)

    if payloads_g0:
        trained_seeds = _parallel_train(
            payloads_g0, cfg,
            train_epochs=cfg.INIT_EPOCHS,
            train_lr=cfg.INIT_LR,
            desc="Gen 0 | Seed Training",
        )
        if trained_seeds:
            population = trained_seeds

    population = pareto_front(population)
    population = fill_with_diversity(population, population, cfg.MIN_POP)
    history[0] = _save_generation_models(population, 0, models_dir)
    _print_summary(0, population)

    # ------------------------------------------------------------------
    # Generations 1 .. GENERATIONS
    # ------------------------------------------------------------------
    for gen in range(1, cfg.GENERATIONS + 1):
        gen_start = time.time()
        logger.info("===== Generation %d =====", gen)

        # Progressive epoch schedule
        if cfg.EPOCH_PROGRESSION:
            child_epochs = cfg.CHILD_EPOCHS + int(gen / 5)
        else:
            child_epochs = cfg.CHILD_EPOCHS

        parent_temp_paths = []   # track for cleanup in finally

        try:
            # --------------------------------------------------------------
            # STEP 1: KDE fit on current population (cheap objectives only)
            # --------------------------------------------------------------
            sampler.fit(population,
                        objective_keys=cfg.CHEAP_OBJECTIVES,
                        generation=gen)

            # --------------------------------------------------------------
            # STEP 2: Sample npc parents (inverse density)
            # --------------------------------------------------------------
            parents = sampler.sample(
                population, cfg.N_CHILDREN, allow_repeats=True
            )
            if not parents:
                logger.warning("Gen %d: sampler returned no parents — skipping", gen)
                history[gen] = history[gen - 1]
                continue

            # --------------------------------------------------------------
            # STEP 3: Generate npc candidate children (NO training yet)
            # --------------------------------------------------------------
            candidates  = []   # list of (child_Individual, parent_Individual, is_approx)

            for p in parents:
                try:
                    new_graph, op_name, target_info = random_operator(p)
                    if new_graph is None:
                        continue

                    approx = is_approx_op(op_name)
                    child  = Individual(new_graph)
                    child.op_name = op_name

                    # --- Lamarckian weight transfer ---
                    try:
                        from morphisms.weights import transfer_weights
                        parent_model = p.build_model()
                        child_model  = child.build_model()
                        transfer_weights(parent_model, child_model,
                                         child.graph, op_name, target_info)
                    except Exception as e:
                        error_logger.error(
                            "Weight transfer failed %s→%s (%s): %s",
                            p.id, child.id, op_name, e
                        )
                        # child still exists with random init; do not discard

                    candidates.append((child, p, approx))

                except Exception as e:
                    error_logger.error("Child generation error from %s: %s", p.id, e)

            if not candidates:
                logger.warning("Gen %d: no candidates generated — skipping", gen)
                history[gen] = history[gen - 1]
                continue

            # --------------------------------------------------------------
            # STEP 4: Evaluate CHEAP objectives on all candidates (free)
            # --------------------------------------------------------------
            valid_candidates = []
            for child, parent, approx in candidates:
                try:
                    cheap = child.evaluate_cheap(
                        objective_keys=cfg.CHEAP_OBJECTIVES,
                        input_size=(1, 3, 32, 32),
                    )
                    param_count = cheap.get("params", 0)
                    if param_count <= cfg.MAX_PARAMS:
                        valid_candidates.append((child, parent, approx))
                    else:
                        logger.debug("Candidate %s exceeds MAX_PARAMS (%d > %d)",
                                     child.id, param_count, cfg.MAX_PARAMS)
                except Exception as e:
                    error_logger.error("Cheap eval failed for candidate %s: %s",
                                       child.id, e)

            if not valid_candidates:
                logger.warning("Gen %d: all candidates exceed MAX_PARAMS", gen)
                history[gen] = history[gen - 1]
                continue

            # --------------------------------------------------------------
            # STEP 5: KDE filter — sample nac accepted children
            #         (inverse density on the CANDIDATES)
            # --------------------------------------------------------------
            candidate_inds = [c for c, _, _ in valid_candidates]
            sampler.fit(candidate_inds,
                        objective_keys=cfg.CHEAP_OBJECTIVES,
                        generation=gen)
            n_accept = min(cfg.N_ACCEPT, len(valid_candidates))
            accepted_inds = sampler.sample(
                candidate_inds, n_accept, allow_repeats=False
            )
            accepted_set = {id(ind) for ind in accepted_inds}

            accepted_triples = [
                (c, p, a) for c, p, a in valid_candidates
                if id(c) in accepted_set
            ]

            logger.info("Gen %d: %d candidates → %d accepted for training",
                        gen, len(valid_candidates), len(accepted_triples))

            # --------------------------------------------------------------
            # STEP 6: Train ONLY accepted children (expensive)
            # --------------------------------------------------------------
            pickled_children = []
            for idx, (child, parent, approx) in enumerate(accepted_triples):
                try:
                    child_sd_path  = None
                    parent_sd_path = None

                    if child.model is not None:
                        child_sd_path = os.path.join(
                            temp_dir, f"child_{child.id}_{gen}.pt"
                        )
                        torch.save(child.model.state_dict(), child_sd_path)

                    if parent.model is not None:
                        parent_sd_path = os.path.join(
                            temp_dir,
                            f"parent_{parent.id}_for_{child.id}_{gen}.pt"
                        )
                        torch.save(parent.model.state_dict(), parent_sd_path)
                        parent_temp_paths.append(parent_sd_path)

                    pc = pickle.dumps((
                        child.graph, child_sd_path,
                        parent.graph, parent_sd_path,
                        approx,
                    ))
                    pickled_children.append((idx, pc))

                except Exception as e:
                    error_logger.error(
                        "Gen %d: serialise error for child %s: %s",
                        gen, child.id, e
                    )

            trained_children = []
            if pickled_children:
                trained_children = _parallel_train(
                    pickled_children, cfg,
                    train_epochs=child_epochs,
                    train_lr=cfg.CHILD_LR,
                    desc=f"Gen {gen} | Training {len(pickled_children)} children",
                )

            # --------------------------------------------------------------
            # STEP 7: Pareto update
            # --------------------------------------------------------------
            if trained_children:
                combined     = population + trained_children
                new_front    = pareto_front(combined)
                new_pop      = fill_with_diversity(new_front, combined,
                                                   cfg.MIN_POP)
                population   = new_pop
            else:
                logger.warning("Gen %d: no children trained successfully", gen)

        except Exception as e:
            error_logger.error("Unhandled error in gen %d: %s\n%s",
                               gen, e, traceback.format_exc())
        finally:
            # Clean up parent temp weight files
            for path in parent_temp_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            elapsed = time.time() - gen_start
            logger.info("Generation %d complete in %.1fs | pop=%d",
                        gen, elapsed, len(population))

        # ------------------------------------------------------------------
        # Save history and model weights for this generation
        # ------------------------------------------------------------------
        history[gen] = _save_generation_models(population, gen, models_dir)
        _print_summary(gen, population)

    return population, history