# morphisms/exact.py
import torch
import torch.nn as nn
from utils.logger import get_logger
from architectures.node import Node
from architectures.graph import ArchitectureGraph
import math

logger = get_logger("morphisms", logfile="logs/morphisms.log")

def _next_node_id(graph: ArchitectureGraph):
    return max(graph.nodes.keys()) + 1 if graph.nodes else 0

def apply_net2deeper(graph: ArchitectureGraph, relu_node_id: int, kernel=1, stride=1, padding=0):
    new_graph = graph.clone()
    if relu_node_id not in new_graph.nodes:
        raise KeyError(f"relu node {relu_node_id} not in graph")

    inferred_ch = None
    old_children = []
    
    # Optimized child lookup
    for nid, node in new_graph.nodes.items():
        if relu_node_id in node.parents:
            old_children.append(nid)
            if 'in_channels' in node.params and inferred_ch is None:
                inferred_ch = node.params['in_channels']

    if inferred_ch is None:
        parent_node = new_graph.nodes[relu_node_id]
        for p in parent_node.parents:
            pn = new_graph.nodes[p]
            if 'out_channels' in pn.params:
                inferred_ch = pn.params['out_channels']
                break

    if inferred_ch is None:
        raise ValueError("Could not infer channel size. Ensure graph nodes expose 'in_channels' or 'out_channels'.")

    conv_id = _next_node_id(new_graph)
    bn_id = conv_id + 1
    relu_new_id = conv_id + 2

    logger.info("Applying Net2Deeper: inserting nodes conv=%d bn=%d relu=%d after relu_node=%d (channels=%d)",
                conv_id, bn_id, relu_new_id, relu_node_id, inferred_ch)

    conv_params = {
        'in_channels': inferred_ch,
        'out_channels': inferred_ch,
        'kernel': kernel,
        'stride': stride,
        'padding': padding,
        'groups': 1
    }
    new_graph.add_node(Node(conv_id, 'conv', conv_params, parents=[relu_node_id]))
    new_graph.add_node(Node(bn_id, 'bn', {'num_features': inferred_ch}, parents=[conv_id]))
    new_graph.add_node(Node(relu_new_id, 'relu', {}, parents=[bn_id]))

    for child_id in old_children:
        child = new_graph.nodes[child_id]
        # List comprehension is faster than manual list appending in Python
        child.parents = [relu_new_id if p == relu_node_id else p for p in child.parents]
        logger.debug("Rewired child %d parents: new_parents=%s", child_id, child.parents)

    return new_graph

def initialize_conv_as_identity(conv_module: nn.Conv2d):
    """
    Initialize 1x1 conv as identity mapping when in_channels == out_channels.
    """
    with torch.no_grad():
        out_c, in_c, kh, kw = conv_module.weight.shape
        if out_c != in_c:
            logger.warning("Identity init requested but out_c != in_c (%d != %d). Skipping identity init.", out_c, in_c)
            return
            
        # Optimized: In-place zeroing instead of allocating a new zeros_like tensor
        conv_module.weight.zero_()
        
        # Optimized: Vectorized assignment bypasses the slow Python 'for i in range(out_c)' loop
        conv_module.weight[range(out_c), range(in_c), kh // 2, kw // 2] = 1.0
        
        if conv_module.bias is not None:
            conv_module.bias.zero_()
            
    logger.info("Initialized Conv module as identity (shape=%s)", tuple(conv_module.weight.shape))

def initialize_bn_as_identity(bn_module: nn.BatchNorm2d):
    with torch.no_grad():
        if hasattr(bn_module, 'weight'):
            bn_module.weight.fill_(1.0)
        if hasattr(bn_module, 'bias'):
            bn_module.bias.zero_()
        bn_module.running_mean.zero_()
        bn_module.running_var.fill_(1.0)
    logger.info("Initialized BatchNorm as identity (num_features=%d)", bn_module.num_features)

def inherit_weights(parent_model: nn.Module, child_model: nn.Module):
    parent_layers = getattr(parent_model, 'layers', None)
    child_layers = getattr(child_model, 'layers', None)
    if parent_layers is None or child_layers is None:
        logger.error("Parent or child model has no 'layers' ModuleDict")
        return

    copied, skipped = 0, 0
    for key in parent_layers.keys():
        if key in child_layers:
            p_sd = parent_layers[key].state_dict()
            c_sd = child_layers[key].state_dict()
            
            # Optimized: Removed `.clone()`
            # load_state_dict safely copies the underlying data. Cloning beforehand 
            # causes a massive 2x VRAM spike during whole-network inheritance.
            to_load = {name: tensor for name, tensor in p_sd.items() 
                       if name in c_sd and tensor.shape == c_sd[name].shape}
            
            if to_load:
                try:
                    child_layers[key].load_state_dict(to_load, strict=False)
                    copied += 1
                    logger.debug("Copied matching params for module %s", key)
                except Exception as e:
                    logger.exception("Failed to load matching params for module %s: %s", key, str(e))
                    skipped += 1
            else:
                skipped += 1
        else:
            skipped += 1
    logger.info("Weight inheritance finished: copied_modules=%d, skipped_modules=%d", copied, skipped)

def apply_net2wider(graph: ArchitectureGraph, conv_node_id: int, widen_by: int = 4):
    new_graph = graph.clone()

    if conv_node_id not in new_graph.nodes:
        raise KeyError(f"Conv node {conv_node_id} not found")

    conv_node = new_graph.nodes[conv_node_id]
    if conv_node.op_type != 'conv':
        raise ValueError("Net2Wider can only be applied to conv nodes")

    old_out = conv_node.params['out_channels']
    new_out = old_out + widen_by

    logger.info("Applying Net2Wider: node=%d old_out=%d new_out=%d", conv_node_id, old_out, new_out)

    conv_node.params['out_channels'] = new_out

    for nid, node in new_graph.nodes.items():
        if conv_node_id in node.parents:
            if node.op_type == 'conv':
                if node.params.get('in_channels') == old_out:
                    node.params['in_channels'] = new_out
                    logger.debug("Updated downstream conv %d in_channels=%d", nid, new_out)
                else:
                    logger.warning("Downstream conv %d in_channels=%s != expected %d", nid, node.params.get('in_channels'), old_out)
            elif node.op_type == 'bn':
                if node.params.get('num_features') == old_out:
                    node.params['num_features'] = new_out
                    logger.debug("Updated downstream BN %d num_features=%d", nid, new_out)
                else:
                    logger.warning("Downstream BN %d num_features=%s != expected %d", nid, node.params.get('num_features'), old_out)
            elif node.op_type not in ('relu', 'identity'):
                logger.warning("Net2Wider: unhandled downstream op %s at node %d", node.op_type, nid)

    return new_graph

def inherit_weights_net2wider(parent_model, child_model, conv_node_id, widen_by):
    key = str(conv_node_id)
    parent_layers = parent_model.layers
    child_layers = child_model.layers

    if key not in parent_layers or key not in child_layers:
        logger.error("Conv node %s missing in parent/child models", key)
        return

    p_conv = parent_layers[key]
    c_conv = child_layers[key]

    with torch.no_grad():
        old_w = p_conv.weight
        old_out = old_w.shape[0]
        new_out = c_conv.weight.shape[0]

        c_conv.weight[:old_out].copy_(old_w)

        # Optimized: Vectorized random sampling.
        # Original used a python loop making individual `c_conv.weight[i].copy_(old_w[src])` calls.
        # This executes instantly in C++ instead of iterating in Python.
        if new_out > old_out:
            src_indices = torch.randint(0, old_out, (new_out - old_out,))
            c_conv.weight[old_out:].copy_(old_w[src_indices])

        if p_conv.bias is not None and c_conv.bias is not None:
            c_conv.bias[:old_out].copy_(p_conv.bias)
            fill_val = p_conv.bias[0] if p_conv.bias.numel() > 0 else 0.0
            c_conv.bias[old_out:].fill_(fill_val)

        logger.info("Net2Wider conv weights copied for node %s (old_out=%d new_out=%d)", key, old_out, new_out)

        for k in parent_layers.keys():
            if k in child_layers:
                p_mod = parent_layers[k]
                c_mod = child_layers[k]

                if isinstance(p_mod, nn.BatchNorm2d) and isinstance(c_mod, nn.BatchNorm2d):
                    p_sd = p_mod.state_dict()
                    c_sd = c_mod.state_dict()
                    new_state = {}
                    
                    for name, tensor in p_sd.items():
                        if name in c_sd:
                            if tensor.shape == c_sd[name].shape:
                                # Optimized: Removed redundant `.clone()`
                                new_state[name] = tensor
                            else:
                                if tensor.ndim == 1 and tensor.shape[0] == old_out and c_sd[name].ndim == 1 and c_sd[name].shape[0] == new_out:
                                    out_vec = c_sd[name].clone()
                                    out_vec[:old_out].copy_(tensor)
                                    if name == 'weight' or name == 'running_var':
                                        out_vec[old_out:].fill_(1.0)
                                    elif name == 'bias' or name == 'running_mean':
                                        out_vec[old_out:].zero_()
                                    new_state[name] = out_vec

                    if new_state:
                        c_mod.load_state_dict(new_state, strict=False)
                        logger.info("Updated BN module %s after widening", k)

def apply_skip_connection(graph: ArchitectureGraph, from_node: int, to_node: int):
    new_graph = graph.clone()

    if from_node == to_node:
        raise ValueError("Skip connection cannot be self-loop")
    if from_node not in new_graph.nodes or to_node not in new_graph.nodes:
        raise ValueError("Invalid node ids for skip connection")
    if from_node in new_graph.nodes[to_node].parents:
        raise ValueError("Skip already exists")

    new_id = _next_node_id(new_graph)

    # Note: original code used kwargs `op="add"`, matching exactly to preserve Graph compatibility
    merge_node = Node(id=new_id, op="add", params={}, parents=[from_node, to_node])

    # Optimized list comprehension for rerouting parents
    for n in new_graph.nodes.values():
        n.parents = [new_id if p == to_node else p for p in n.parents]

    new_graph.add_node(merge_node)

    if new_graph.output_node == to_node:
        new_graph.set_output(new_id)

    return new_graph