# models/cells.py
from architectures.node import Node

def add_normal_cell(g, node_counter, input_id, channels):
    """
    Adds a minimal Normal Cell to the graph. 
    Preserves spatial resolution and channel count.
    """
    conv_id = node_counter
    g.add_node(Node(conv_id, 'conv', {
        'in_channels': channels, 'out_channels': channels,
        'kernel_size': 3, 'padding': 1
    }, parents=[input_id]))
    
    bn_id = node_counter + 1
    g.add_node(Node(bn_id, 'bn', {'num_features': channels}, parents=[conv_id]))
    
    relu_id = node_counter + 2
    g.add_node(Node(relu_id, 'relu', {}, parents=[bn_id]))
    
    return g, relu_id, node_counter + 3

def add_reduction_cell(g, node_counter, input_id, in_channels, out_channels):
    """
    Adds a minimal Reduction Cell to the graph.
    Halves spatial resolution (via maxpool) and increases channel count.
    """
    pool_id = node_counter
    g.add_node(Node(pool_id, 'maxpool', {'kernel_size': 2}, parents=[input_id]))
    
    conv_id = node_counter + 1
    g.add_node(Node(conv_id, 'conv', {
        'in_channels': in_channels, 'out_channels': out_channels,
        'kernel_size': 3, 'padding': 1
    }, parents=[pool_id]))
    
    bn_id = node_counter + 2
    g.add_node(Node(bn_id, 'bn', {'num_features': out_channels}, parents=[conv_id]))
    
    relu_id = node_counter + 3
    g.add_node(Node(relu_id, 'relu', {}, parents=[bn_id]))
    
    return g, relu_id, node_counter + 4