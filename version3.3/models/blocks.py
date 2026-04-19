# models/blocks.py
# =============================================================================
# Low-level graph-node building blocks.
#
# These work correctly for ANY spatial dimension — the CompiledModel's
# shape-inference engine resolves actual channel/spatial sizes at build time.
# All convolutions use 3x3 with padding=1 so they preserve spatial dimensions
# (stride controls downsampling, not kernel size).
# =============================================================================


def add_conv_block(builder, C_in, C_out, kernel_size, stride, padding, parent_id):
    """
    Conv → BatchNorm → ReLU sequence as graph nodes.
    Works for any input spatial size ≥ kernel_size.
    """
    parents = [parent_id] if parent_id is not None else []

    conv = builder.add_node('conv', {
        'in_channels':  C_in,
        'out_channels': C_out,
        'kernel_size':  kernel_size,
        'stride':       stride,
        'padding':      padding,
    }, parents)

    bn   = builder.add_node('bn',   {'num_features': C_out}, [conv])
    relu = builder.add_node('relu', {}, [bn])

    return relu


def add_res_block(builder, C_in, C_out, stride, parent_id):
    """
    Residual block: two 3x3 convs with a learnable shortcut projection
    when channels or spatial dimensions change.

    Structure:
        input
          ├─ Conv(3x3) → BN → ReLU → Conv(3x3) → BN ──┐
          └─ [projection 1x1 if C_in≠C_out or stride≠1] │
                                                    Add → ReLU → output
    """
    # --- Main path ---
    conv1 = builder.add_node('conv', {
        'in_channels': C_in, 'out_channels': C_out,
        'kernel_size': 3,    'stride': stride, 'padding': 1,
    }, [parent_id])
    bn1   = builder.add_node('bn',   {'num_features': C_out}, [conv1])
    relu1 = builder.add_node('relu', {},                      [bn1])

    conv2 = builder.add_node('conv', {
        'in_channels': C_out, 'out_channels': C_out,
        'kernel_size': 3,     'stride': 1, 'padding': 1,
    }, [relu1])
    bn2 = builder.add_node('bn', {'num_features': C_out}, [conv2])

    # --- Shortcut (projection if dimensions change) ---
    shortcut_id = parent_id
    if stride != 1 or C_in != C_out:
        s_conv      = builder.add_node('conv', {
            'in_channels': C_in, 'out_channels': C_out,
            'kernel_size': 1,   'stride': stride, 'padding': 0,
        }, [parent_id])
        s_bn        = builder.add_node('bn', {'num_features': C_out}, [s_conv])
        shortcut_id = s_bn

    # --- Merge ---
    add_node = builder.add_node('add',  {},  [bn2, shortcut_id])
    relu_out = builder.add_node('relu', {}, [add_node])

    return relu_out

# # models/blocks.py
# # =============================================================================
# # Low-level graph-node building blocks.
# # These work correctly for ANY spatial dimension — the CompiledModel's
# # shape-inference engine resolves actual channel/spatial sizes at build time.
# # All convolutions use 3x3 with padding=1 so they preserve spatial dimensions
# # (stride controls downsampling, not kernel size).
# # =============================================================================


# def add_conv_block(builder, C_in, C_out, kernel_size, stride, padding, parent_id):
#     """
#     Conv → BatchNorm → ReLU sequence as graph nodes.
#     Works for any input spatial size ≥ kernel_size.
#     """
#     parents = [parent_id] if parent_id is not None else []

#     conv = builder.add_node('conv', {
#         'in_channels':  C_in,
#         'out_channels': C_out,
#         'kernel_size':  kernel_size,
#         'stride':       stride,
#         'padding':      padding,
#     }, parents)

#     bn   = builder.add_node('bn',   {'num_features': C_out}, [conv])
#     relu = builder.add_node('relu', {}, [bn])

#     return relu


# def add_res_block(builder, C_in, C_out, stride, parent_id):
#     """
#     Residual block: two 3x3 convs with a learnable shortcut projection
#     when channels or spatial dimensions change.

#     Structure:
#         input
#           ├─ Conv(3x3) → BN → ReLU → Conv(3x3) → BN ──┐
#           └─ [projection 1x1 if C_in≠C_out or stride≠1] │
#                                                     Add → ReLU → output
#     """
#     # --- Main path ---
#     conv1 = builder.add_node('conv', {
#         'in_channels': C_in, 'out_channels': C_out,
#         'kernel_size': 3,    'stride': stride, 'padding': 1,
#     }, [parent_id])
#     bn1   = builder.add_node('bn',   {'num_features': C_out}, [conv1])
#     relu1 = builder.add_node('relu', {},                      [bn1])

#     conv2 = builder.add_node('conv', {
#         'in_channels': C_out, 'out_channels': C_out,
#         'kernel_size': 3,     'stride': 1, 'padding': 1,
#     }, [relu1])
#     bn2 = builder.add_node('bn', {'num_features': C_out}, [conv2])

#     # --- Shortcut (projection if dimensions change) ---
#     shortcut_id = parent_id
#     if stride != 1 or C_in != C_out:
#         s_conv      = builder.add_node('conv', {
#             'in_channels': C_in, 'out_channels': C_out,
#             'kernel_size': 1,    'stride': stride, 'padding': 0,
#         }, [parent_id])
#         s_bn        = builder.add_node('bn', {'num_features': C_out}, [s_conv])
#         shortcut_id = s_bn

#     # --- Merge ---
#     add_node = builder.add_node('add',  {},  [bn2, shortcut_id])
#     relu_out = builder.add_node('relu', {}, [add_node])

#     return relu_out