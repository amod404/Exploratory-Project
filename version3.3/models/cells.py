# models/cells.py
# =============================================================================
# Cell-level building blocks composed from blocks.py primitives.
#
# A "cell" is the repeatable unit of the architecture body.
# A "reduction cell" halves spatial dimensions while doubling channels.
#
# Both cells work correctly for any input spatial size because the underlying
# Conv/BN layers handle the actual tensor shapes at CompiledModel build time.
# =============================================================================

from models.blocks import add_conv_block, add_res_block


def add_cell(builder, C_in, C_out, stride, parent_id):
    """
    Normal Cell: optional channel/spatial adaptation → residual block.

    If C_in == C_out and stride == 1 (no dimension change) the adaptation
    1x1 conv is skipped, saving parameters.

    Used for feature extraction at a fixed spatial resolution.
    Compatible with: 32x32 (CIFAR, GTSRB), 64x64 (Tiny-ImageNet, EuroSAT stem output),
                     56x56 / 28x28 / 14x14 (ImageNet-style bodies).
    """
    if C_in != C_out or stride != 1:
        # 1x1 conv to adapt channel count / spatial size before the res block
        parent_id = add_conv_block(
            builder, C_in, C_out,
            kernel_size=1, stride=stride, padding=0,
            parent_id=parent_id,
        )

    return add_res_block(builder, C_out, C_out, stride=1, parent_id=parent_id)


def add_reduction_cell(builder, C_in, C_out, parent_id):
    """
    Reduction Cell: stride-2 Conv-BN-ReLU that halves spatial resolution
    and optionally changes channel count.

    A plain stride-2 conv is used rather than a pooling op so the NAS
    morphism operators (prune, widen, sepconv) can mutate it freely.
    """
    return add_conv_block(
        builder, C_in, C_out,
        kernel_size=3, stride=2, padding=1,
        parent_id=parent_id,
    )


# # models/cells.py
# # =============================================================================
# # Cell-level building blocks composed from blocks.py primitives.
# #
# # A "cell" is the repeatable unit of the architecture body.
# # A "reduction cell" halves spatial dimensions while doubling channels.
# #
# # Both cells work correctly for any input spatial size because the underlying
# # Conv/BN layers handle the actual tensor shapes at CompiledModel build time.
# # =============================================================================

# from models.blocks import add_conv_block, add_res_block


# def add_cell(builder, C_in, C_out, stride, parent_id):
#     """
#     Normal Cell: optional channel/spatial adaptation → residual block.

#     If C_in == C_out and stride == 1 (no dimension change) the adaptation
#     1x1 conv is skipped, saving parameters.

#     Used for feature extraction at a fixed spatial resolution.
#     Compatible with: 32x32 (CIFAR), 64x64 (Tiny-ImageNet stem output),
#                      56x56 / 28x28 / 14x14 (ImageNet-style bodies).
#     """
#     if C_in != C_out or stride != 1:
#         # 1x1 conv to adapt channel count / spatial size before the res block
#         parent_id = add_conv_block(
#             builder, C_in, C_out,
#             kernel_size=1, stride=stride, padding=0,
#             parent_id=parent_id,
#         )

#     return add_res_block(builder, C_out, C_out, stride=1, parent_id=parent_id)


# def add_reduction_cell(builder, C_in, C_out, parent_id):
#     """
#     Reduction Cell: stride-2 Conv-BN-ReLU that halves spatial resolution
#     and optionally changes channel count.

#     A plain stride-2 conv is used rather than a pooling op so the NAS
#     morphism operators (prune, widen, sepconv) can mutate it freely.
#     """
#     return add_conv_block(
#         builder, C_in, C_out,
#         kernel_size=3, stride=2, padding=1,
#         parent_id=parent_id,
#     )