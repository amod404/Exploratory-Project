# architectures/compiler.py
import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("compiler", logfile="logs/compiler.log")


class CompiledModel(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.layers = nn.ModuleDict()
        logger.info("Initializing CompiledModel")
        
        # Pre-computed execution plan to avoid O(V+E) topological sorts and string ops in forward()
        self._execution_plan = []
        self._output_node_id = None
        
        self._build()

    def _build(self):
        logger.info("Building modules for graph with %d nodes", len(self.graph.nodes))

        for node_id, node in self.graph.nodes.items():
            op = node.op_type.lower()
            key = str(node_id)

            try:
                # ---------------- Conv ----------------
                if op == 'conv':
                    self.layers[key] = nn.Conv2d(
                        node.params['in_channels'],
                        node.params['out_channels'],
                        kernel_size=node.params.get('kernel_size', 3),
                        stride=node.params.get('stride', 1),
                        padding=node.params.get('padding', 1),
                        bias=False,
                        groups=node.params.get('groups', 1)
                    )
                    logger.debug("Created Conv2d node %s: in=%d out=%d", key, node.params['in_channels'], node.params['out_channels'])

                # ---------------- Separable Conv ----------------
                elif op in ('sep_conv', 'separableconv2d'):
                    in_c = node.params['in_channels']
                    out_c = node.params['out_channels']
                    k = node.params.get('kernel_size', 3)
                    stride = node.params.get('stride', 1)
                    pad = node.params.get('padding', 1)

                    self.layers[key] = nn.Sequential(
                        nn.Conv2d(in_c, in_c, kernel_size=k, stride=stride, padding=pad, groups=in_c, bias=False),
                        nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                    )
                    logger.debug("Created SeparableConv node %s: in=%d out=%d", key, in_c, out_c)

                # ---------------- BatchNorm ----------------
                elif op == 'bn':
                    self.layers[key] = nn.BatchNorm2d(node.params['num_features'])
                    logger.debug("Created BN node %s: features=%d", key, node.params['num_features'])

                # ---------------- ReLU ----------------
                elif op == 'relu':
                    self.layers[key] = nn.ReLU(inplace=True)
                    logger.debug("Created ReLU node %s", key)

                # ---------------- Flatten ----------------
                elif op == 'flatten':
                    self.layers[key] = nn.Flatten(start_dim=1)
                    logger.debug("Created Flatten node %s", key)

                # ---------------- Linear / FC ----------------
                elif op in ('fc', 'linear'):
                    self.layers[key] = nn.Linear(node.params['in_features'], node.params['out_features'])
                    logger.debug("Created Linear node %s: in=%d out=%d", key, node.params['in_features'], node.params['out_features'])

                # ---------------- Identity & Merge ----------------
                elif op in ('identity', 'add', 'concat'):
                    self.layers[key] = nn.Identity()
                    logger.debug("Created %s node %s", op, key)

                # ---------------- Unknown ----------------
                else:
                    logger.warning("Unknown op_type '%s' for node %s — creating Identity placeholder", op, key)
                    self.layers[key] = nn.Identity()

            except Exception:
                logger.exception("Failed creating module for node %s op=%s params=%s", key, op, node.params)
                raise

        # Compile the static execution plan so forward pass is O(N)
        self._compile_execution_plan()

    def _compile_execution_plan(self):
        order = self.graph.topological_sort()
        for node_id in order:
            node = self.graph.nodes[node_id]
            self._execution_plan.append((
                node_id, 
                str(node_id), 
                node.parents, 
                node.op_type.lower()
            ))
        self._output_node_id = self.graph.output_node

    def forward(self, x):
        cache = {}

        for node_id, layer_key, parents, op in self._execution_plan:
            # -------- Input Gathering --------
            if not parents:
                inp = x
                logger.debug("Node %s has no parents — using graph input", node_id)
            else:
                try:
                    if len(parents) == 1:
                        inp = cache[parents[0]]
                    else:
                        tensors = [cache[p] for p in parents]

                        if op == 'add':
                            try:
                                # Optimized: iterative addition avoids O(N * tensor_size) memory allocation of torch.stack
                                inp = tensors[0]
                                for t in tensors[1:]:
                                    inp = inp + t
                            except Exception:
                                logger.exception("Add merge failed for node %s with parent shapes: %s", node_id, [t.shape for t in tensors])
                                raise

                        else:
                            # Default to concat for 'concat' or unknown operations
                            try:
                                inp = torch.cat(tensors, dim=1)
                            except Exception:
                                logger.exception("%s merge failed for node %s with parent shapes: %s", "Concat" if op == 'concat' else "Default concat", node_id, [t.shape for t in tensors])
                                raise
                            if op != 'concat':
                                logger.debug("Multiple parents for node %s but op=%s — defaulted to concat", node_id, op)
                
                except KeyError:
                    # Fallback lookup to provide exact missing keys for logging (EAFP pattern)
                    missing = [p for p in parents if p not in cache]
                    logger.error("Node %s parent(s) %s not in cache. Available keys: %s", node_id, missing, list(cache.keys()))
                    raise KeyError(f"Missing parents for node {node_id}: {missing}")

            # -------- Apply Layer --------
            try:
                layer = self.layers[layer_key]
            except KeyError:
                logger.error("No layer registered for node %s. Available layers: %s", node_id, list(self.layers.keys()))
                raise KeyError(node_id)

            try:
                # Dynamic auto-flattening and Linear recreation
                if isinstance(layer, nn.Linear):
                    if inp.dim() > 2:
                        inp = inp.view(inp.size(0), -1)
                        logger.debug("Auto-flattened input for Linear at node %s -> shape %s", node_id, tuple(inp.shape))

                    expected_in = layer.in_features
                    actual_in = inp.size(1)
                    if expected_in != actual_in:
                        logger.info("Linear shape mismatch at node %s: expected %d got %d — recreating Linear(%d -> %d)", 
                                    node_id, expected_in, actual_in, actual_in, layer.out_features)
                        new_linear = nn.Linear(actual_in, layer.out_features)
                        try:
                            new_linear = new_linear.to(layer.weight.device)
                        except Exception:
                            pass
                        self.layers[layer_key] = new_linear
                        layer = new_linear

                out = layer(inp)

            except Exception as exc:
                logger.exception("Layer forward failed at node %s op=%s inp_shape=%s", node_id, op, getattr(inp, 'shape', None))
                raise

            cache[node_id] = out
            logger.debug("Node %s produced output shape %s", node_id, tuple(out.shape))

        try:
            return cache[self._output_node_id]
        except KeyError:
            logger.error("Output node %s not computed. Cache keys: %s", self._output_node_id, list(cache.keys()))
            raise KeyError("Output not computed")