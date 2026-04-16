# evolution/individual.py
import uuid
import torch
from architectures.compiler import CompiledModel
from utils.logger import get_logger

logger = get_logger("individual", logfile="logs/individual.log")


class Individual:
    def __init__(self, graph):
        self.id       = uuid.uuid4().hex[:8]
        self.graph    = graph
        self.model    = None        # set by build_model(); cleared before pickling
        self.f_cheap  = None        # dict: e.g. {"params": ..., "flops": ...}
        self.f_exp    = None        # dict: {"val_error": float}
        self.op_name  = None        # set by lemonade loop
        logger.debug("Created Individual %s", self.id)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def build_model(self, input_shape=(1, 3, 32, 32)) -> CompiledModel:
        """
        Build (once) or return the cached CompiledModel.

        IMPORTANT:
        - Always pass the correct input_shape for your dataset.
          Use cfg.INPUT_SIZE everywhere, e.g.:
              ind.build_model(input_shape=cfg.INPUT_SIZE)
        - Do NOT call build_model() multiple times expecting different objects.
          After the first call, self.model is cached and the same object is
          returned every time (regardless of input_shape).
        """
        if self.model is None:
            logger.debug("Building CompiledModel for Individual %s", self.id)
            self.model = CompiledModel(self.graph, input_shape=input_shape)
        return self.model

    # ------------------------------------------------------------------
    # Cheap objectives
    # ------------------------------------------------------------------

    def evaluate_cheap(self, objective_keys=("params", "flops"),
                       input_size=(1, 3, 32, 32)):
        """
        Compute cheap objectives (cached after first call).

        "params"  → fast graph-based count, NO model build required
        "flops"   → builds model + runs thop (slower)

        Always pass cfg.INPUT_SIZE as input_size for correctness.
        """
        if self.f_cheap is not None:
            return self.f_cheap

        result = {}

        if "params" in objective_keys:
            from objectives.cheap import count_params_from_graph
            result["params"] = count_params_from_graph(self.graph)

        if "flops" in objective_keys:
            from objectives.cheap import estimate_flops
            model = self.build_model(input_shape=input_size)
            model.eval()
            with torch.no_grad():
                result["flops"] = estimate_flops(model, input_size=input_size)

        self.f_cheap = result
        logger.debug("Individual %s cheap: %s", self.id, self.f_cheap)
        return self.f_cheap

    # ------------------------------------------------------------------
    # Expensive objective
    # ------------------------------------------------------------------

    def evaluate_expensive(self, train_loader, val_loader,
                           device="cpu", epochs=1, lr=0.01,
                           weight_decay=1e-4, optimizer_name="sgd",
                           show_progress=True):
        """
        Train + evaluate.  Called directly only in serial fallback; the normal
        LEMONADE loop uses the worker function in lemonade_full.py instead.
        """
        if self.f_exp is not None:
            return self.f_exp

        model = self.build_model()
        try:
            from objectives.expensive import evaluate_on_data
            val_error = evaluate_on_data(
                model, train_loader, val_loader,
                device=device, epochs=epochs, lr=lr,
                weight_decay=weight_decay, optimizer_name=optimizer_name,
                show_progress=show_progress,
            )
            self.f_exp = {"val_error": val_error}
        finally:
            if self.model is not None:
                self.model.cpu()
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.f_exp