# models/base_net.py
from architectures.graph import ArchitectureGraph
from architectures.node import Node
from models.cells import add_normal_cell, add_reduction_cell

def build_sequential_macro_architecture(
    in_channels=3,
    init_channels=16,
    num_classes=10,
    num_cells=5,
    image_size=32
):
    """
    Implements the simpler macro architecture scheme from the LEMONADE paper:
    Sequentially stacking cells.
    """
    g = ArchitectureGraph()
    node_counter = 0
    
    # 1. Stem (Initial feature extraction)
    stem_conv_id = node_counter
    g.add_node(Node(stem_conv_id, 'conv', {
        'in_channels': in_channels, 'out_channels': init_channels,
        'kernel_size': 3, 'padding': 1
    }, parents=[]))
    node_counter += 1
    
    current_input = stem_conv_id
    current_channels = init_channels
    
    # 2. Sequentially Stack Cells
    # Place reduction cells at ~33% and ~66% of the network depth
    reduction_indices = [num_cells // 3, (2 * num_cells) // 3]
    
    for i in range(1, num_cells + 1):
        if i in reduction_indices:
            # Add Reduction Cell
            out_channels = current_channels * 2
            g, current_input, node_counter = add_reduction_cell(
                g, node_counter, current_input, current_channels, out_channels
            )
            current_channels = out_channels
        else:
            # Add Normal Cell
            g, current_input, node_counter = add_normal_cell(
                g, node_counter, current_input, current_channels
            )
            
    # 3. Head (Flatten + Linear)
    flatten_id = node_counter
    g.add_node(Node(flatten_id, 'flatten', {}, parents=[current_input]))
    node_counter += 1
    
    # Calculate features for the Linear layer based on how many reductions occurred
    num_reductions = len(reduction_indices)
    spatial_dim = image_size // (2 ** num_reductions)
    in_features = current_channels * (spatial_dim ** 2)
    
    linear_id = node_counter
    g.add_node(Node(linear_id, 'linear', {
        'in_features': in_features,
        'out_features': num_classes
    }, parents=[flatten_id]))
    
    g.set_output(linear_id)
    
    return g