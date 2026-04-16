# models/basenet.py
# =============================================================================
# Builds the seed (base) architecture graph for each supported dataset.
#
# Architecture layout per dataset:
#
#  CIFAR-10 / CIFAR-100 / CINIC-10 / GTSRB (input 32x32)
#  ─────────────────────────────────────────────────────
#  Stem : Conv(3→16, 3x3, s=1) → BN → ReLU          [32x32]
#  Body : 3 stages, 2 reduction cells
#         Stage 0: Cell(16,16)  → Reduce(16→32,s=2)  [32→16]
#         Stage 1: Cell(32,32)  → Reduce(32→64,s=2)  [16→8]
#         Stage 2: Cell(64,64)                       [8x8]
#  Head : MaxPool(8×8) → Flatten → Linear(64, C)
#
#  TINY_IMAGENET / EUROSAT (input 64x64)
#  ─────────────────────────────────────
#  Stem : Conv(3→16, 3x3, s=2) → BN → ReLU          [32x32]
#  Body : same 3-stage layout as CIFAR                [32→16→8]
#  Head : MaxPool(8×8) → Flatten → Linear(64, C)
#
#  IMAGENET / PLACES365 / COCO  (input 224x224)
#  ────────────────────────────────────────────
#  Stem : Conv(3→32, 7x7, s=2) → BN → ReLU → MaxPool(3,s=2)  [56x56]
#  Body : 3 stages, 2 reduction cells
#         Stage 0: Cell(32,32)   → Reduce(32→64,s=2)  [56→28]
#         Stage 1: Cell(64,64)   → Reduce(64→128,s=2) [28→14]
#         Stage 2: Cell(128,128)                        [14x14]
#  Head : MaxPool(14×14) → Flatten → Linear(128, C)
#
# The init_channels and final_spatial_size values are chosen so that
# MaxPool perfectly collapses the feature map to 1x1.
# =============================================================================

from architectures.node import Node
from architectures.graph import ArchitectureGraph
from models.cells import add_cell, add_reduction_cell


# ---------------------------------------------------------------------------
# Groups of datasets that share the same architecture layout
# ---------------------------------------------------------------------------
_CIFAR_DATASETS         = {"CIFAR-10", "CIFAR-100", "CINIC-10", "GTSRB"}
_TINY_IMAGENET_DATASETS = {"TINY_IMAGENET", "EUROSAT"}
_LARGE_DATASETS         = {"IMAGENET", "PLACES365", "COCO"}


class GraphBuilder:
    """
    Helper that assigns sequential node IDs and keeps a pointer to the last
    added node, making it easy to build linear or branching architectures.
    """
    def __init__(self):
        self.g          = ArchitectureGraph()
        self.node_count = 0

    def add_node(self, op_type, params, parents):
        node = Node(self.node_count, op_type, params, parents)
        self.g.add_node(node)
        self.node_count += 1
        return node.id


def build_basenet_graph(num_classes: int, dataset_type: str) -> ArchitectureGraph:
    """
    Construct the seed architecture graph for *dataset_type*.

    Parameters
    ----------
    num_classes  : number of output classes (from cfg.NUM_CLASSES)
    dataset_type : one of the supported dataset name strings

    Returns
    -------
    ArchitectureGraph ready to be compiled into a CompiledModel
    """
    builder = GraphBuilder()

    # -----------------------------------------------------------------------
    # 1. Stem — dataset-specific downsampling to a common feature-map size
    # -----------------------------------------------------------------------
    if dataset_type in _CIFAR_DATASETS:
        # Input: 32x32  →  stem output: 32x32  (no downsampling)
        init_channels = 16
        curr = builder.add_node('conv', {
            'in_channels': 3, 'out_channels': init_channels,
            'kernel_size': 3, 'stride': 1, 'padding': 1,
        }, [])
        curr = builder.add_node('bn',   {'num_features': init_channels}, [curr])
        curr = builder.add_node('relu', {}, [curr])

    elif dataset_type in _TINY_IMAGENET_DATASETS:
        # Input: 64x64  →  stem output: 32x32  (single stride-2 conv)
        init_channels = 16
        curr = builder.add_node('conv', {
            'in_channels': 3, 'out_channels': init_channels,
            'kernel_size': 3, 'stride': 2, 'padding': 1,
        }, [])
        curr = builder.add_node('bn',   {'num_features': init_channels}, [curr])
        curr = builder.add_node('relu', {}, [curr])

    elif dataset_type in _LARGE_DATASETS:
        # Input: 224x224  →  stem output: 56x56  (7x7 s=2 + maxpool s=2)
        # Use 32 init channels (vs 16 for small datasets) for richer features
        # on high-resolution inputs.
        init_channels = 32
        curr = builder.add_node('conv', {
            'in_channels': 3, 'out_channels': init_channels,
            'kernel_size': 7, 'stride': 2, 'padding': 3,
        }, [])
        curr = builder.add_node('bn',   {'num_features': init_channels}, [curr])
        curr = builder.add_node('relu', {}, [curr])
        curr = builder.add_node('maxpool', {'kernel_size': 3, 'stride': 2}, [curr])

    else:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. "
            f"Supported: CIFAR-10, CIFAR-100, CINIC-10, GTSRB, TINY_IMAGENET, EUROSAT, IMAGENET, PLACES365, COCO"
        )

    # -----------------------------------------------------------------------
    # 2. Body — 3 stages of cells + 2 reduction cells (shared structure)
    #
    #  All datasets use the SAME body structure after the stem, because the
    #  stem normalises spatial dimensions to a common starting point:
    #    CIFAR / Tiny-IN / EUROSAT →  32x32  →  body outputs  8x8   (C=64)
    #    Large datasets            →  56x56  →  body outputs 14x14  (C=128)
    # -----------------------------------------------------------------------
    C_curr = init_channels
    num_cells_per_stage = 1   # kept small for fast NAS search; operators grow it

    for stage in range(3):
        for _ in range(num_cells_per_stage):
            curr = add_cell(builder, C_curr, C_curr, stride=1, parent_id=curr)

        if stage < 2:        # reduction after stages 0 and 1 only
            C_next = C_curr * 2
            curr   = add_reduction_cell(builder, C_curr, C_next, parent_id=curr)
            C_curr = C_next

    # -----------------------------------------------------------------------
    # 3. Head — global pooling + classifier
    #
    #  After the body:
    #    CIFAR / Tiny-IN / EUROSAT →  8x8    →  maxpool(8×8)   → 1x1
    #    Large datasets            →  14x14  →  maxpool(14×14) → 1x1
    # -----------------------------------------------------------------------
    if dataset_type in _CIFAR_DATASETS or dataset_type in _TINY_IMAGENET_DATASETS:
        final_spatial = 8
    else:
        final_spatial = 14

    curr     = builder.add_node('maxpool', {
        'kernel_size': final_spatial, 'stride': final_spatial,
    }, [curr])
    curr     = builder.add_node('flatten', {}, [curr])
    out_node = builder.add_node('linear',  {
        'in_features': C_curr, 'out_features': num_classes,
    }, [curr])

    builder.g.set_output(out_node)
    return builder.g

# # models/basenet.py
# # =============================================================================
# # Builds the seed (base) architecture graph for each supported dataset.
# #
# # Architecture layout per dataset:
# #
# #  CIFAR-10 / CIFAR-100  (input 32x32)
# #  ─────────────────────
# #  Stem : Conv(3→16, 3x3, s=1) → BN → ReLU          [32x32]
# #  Body : 3 stages, 2 reduction cells
# #         Stage 0: Cell(16,16)  → Reduce(16→32,s=2)  [32→16]
# #         Stage 1: Cell(32,32)  → Reduce(32→64,s=2)  [16→8]
# #         Stage 2: Cell(64,64)                         [8x8]
# #  Head : MaxPool(8×8) → Flatten → Linear(64, C)
# #
# #  TINY_IMAGENET  (input 64x64, 200 classes)
# #  ─────────────
# #  Stem : Conv(3→16, 3x3, s=2) → BN → ReLU          [32x32]
# #  Body : same 3-stage layout as CIFAR                [32→16→8]
# #  Head : MaxPool(8×8) → Flatten → Linear(64, 200)
# #
# #  IMAGENET / PLACES365 / COCO  (input 224x224)
# #  ──────────────────────────────
# #  Stem : Conv(3→32, 7x7, s=2) → BN → ReLU → MaxPool(3,s=2)  [56x56]
# #  Body : 3 stages, 2 reduction cells
# #         Stage 0: Cell(32,32)   → Reduce(32→64,s=2)  [56→28]
# #         Stage 1: Cell(64,64)   → Reduce(64→128,s=2) [28→14]
# #         Stage 2: Cell(128,128)                        [14x14]
# #  Head : MaxPool(14×14) → Flatten → Linear(128, C)
# #
# # The init_channels and final_spatial_size values are chosen so that
# # MaxPool perfectly collapses the feature map to 1x1.
# # =============================================================================

# from architectures.node import Node
# from architectures.graph import ArchitectureGraph
# from models.cells import add_cell, add_reduction_cell


# # ---------------------------------------------------------------------------
# # Groups of datasets that share the same architecture layout
# # ---------------------------------------------------------------------------
# _CIFAR_DATASETS        = {"CIFAR-10", "CIFAR-100"}
# _TINY_IMAGENET_DATASETS = {"TINY_IMAGENET"}
# _LARGE_DATASETS        = {"IMAGENET", "PLACES365", "COCO"}


# class GraphBuilder:
#     """
#     Helper that assigns sequential node IDs and keeps a pointer to the last
#     added node, making it easy to build linear or branching architectures.
#     """
#     def __init__(self):
#         self.g          = ArchitectureGraph()
#         self.node_count = 0

#     def add_node(self, op_type, params, parents):
#         node = Node(self.node_count, op_type, params, parents)
#         self.g.add_node(node)
#         self.node_count += 1
#         return node.id


# def build_basenet_graph(num_classes: int, dataset_type: str) -> ArchitectureGraph:
#     """
#     Construct the seed architecture graph for *dataset_type*.

#     Parameters
#     ----------
#     num_classes  : number of output classes (from cfg.NUM_CLASSES)
#     dataset_type : one of the supported dataset name strings

#     Returns
#     -------
#     ArchitectureGraph ready to be compiled into a CompiledModel
#     """
#     builder = GraphBuilder()

#     # -----------------------------------------------------------------------
#     # 1. Stem — dataset-specific downsampling to a common feature-map size
#     # -----------------------------------------------------------------------
#     if dataset_type in _CIFAR_DATASETS:
#         # Input: 32x32  →  stem output: 32x32  (no downsampling)
#         init_channels = 16
#         curr = builder.add_node('conv', {
#             'in_channels': 3, 'out_channels': init_channels,
#             'kernel_size': 3, 'stride': 1, 'padding': 1,
#         }, [])
#         curr = builder.add_node('bn',   {'num_features': init_channels}, [curr])
#         curr = builder.add_node('relu', {}, [curr])

#     elif dataset_type in _TINY_IMAGENET_DATASETS:
#         # Input: 64x64  →  stem output: 32x32  (single stride-2 conv)
#         init_channels = 16
#         curr = builder.add_node('conv', {
#             'in_channels': 3, 'out_channels': init_channels,
#             'kernel_size': 3, 'stride': 2, 'padding': 1,
#         }, [])
#         curr = builder.add_node('bn',   {'num_features': init_channels}, [curr])
#         curr = builder.add_node('relu', {}, [curr])

#     elif dataset_type in _LARGE_DATASETS:
#         # Input: 224x224  →  stem output: 56x56  (7x7 s=2 + maxpool s=2)
#         # Use 32 init channels (vs 16 for small datasets) for richer features
#         # on high-resolution inputs.
#         init_channels = 32
#         curr = builder.add_node('conv', {
#             'in_channels': 3, 'out_channels': init_channels,
#             'kernel_size': 7, 'stride': 2, 'padding': 3,
#         }, [])
#         curr = builder.add_node('bn',   {'num_features': init_channels}, [curr])
#         curr = builder.add_node('relu', {}, [curr])
#         curr = builder.add_node('maxpool', {'kernel_size': 3, 'stride': 2}, [curr])

#     else:
#         raise ValueError(
#             f"Unknown dataset_type '{dataset_type}'. "
#             f"Supported: CIFAR-10, CIFAR-100, TINY_IMAGENET, IMAGENET, PLACES365, COCO"
#         )

#     # -----------------------------------------------------------------------
#     # 2. Body — 3 stages of cells + 2 reduction cells (shared structure)
#     #
#     #  All datasets use the SAME body structure after the stem, because the
#     #  stem normalises spatial dimensions to a common starting point:
#     #    CIFAR / Tiny-IN  →  32x32  →  body outputs  8x8   (C=64)
#     #    Large datasets   →  56x56  →  body outputs 14x14  (C=128)
#     # -----------------------------------------------------------------------
#     C_curr = init_channels
#     num_cells_per_stage = 1   # kept small for fast NAS search; operators grow it

#     for stage in range(3):
#         for _ in range(num_cells_per_stage):
#             curr = add_cell(builder, C_curr, C_curr, stride=1, parent_id=curr)

#         if stage < 2:        # reduction after stages 0 and 1 only
#             C_next = C_curr * 2
#             curr   = add_reduction_cell(builder, C_curr, C_next, parent_id=curr)
#             C_curr = C_next

#     # -----------------------------------------------------------------------
#     # 3. Head — global pooling + classifier
#     #
#     #  After the body:
#     #    CIFAR / Tiny-IN   →  8x8    →  maxpool(8×8)   → 1x1
#     #    Large datasets    →  14x14  →  maxpool(14×14) → 1x1
#     # -----------------------------------------------------------------------
#     if dataset_type in _CIFAR_DATASETS or dataset_type in _TINY_IMAGENET_DATASETS:
#         final_spatial = 8
#     else:
#         final_spatial = 14

#     curr     = builder.add_node('maxpool', {
#         'kernel_size': final_spatial, 'stride': final_spatial,
#     }, [curr])
#     curr     = builder.add_node('flatten', {}, [curr])
#     out_node = builder.add_node('linear',  {
#         'in_features': C_curr, 'out_features': num_classes,
#     }, [curr])

#     builder.g.set_output(out_node)
#     return builder.g