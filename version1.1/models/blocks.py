# models/blocks.py\
#not needed
'''
from models.cells import add_normal_cell, add_reduction_cell

def add_block(g, node_counter, input_id, num_normal_cells, in_channels, is_reduction=True):
    """
    Adds a Block consisting of `num_normal_cells` Normal Cells.
    If `is_reduction` is True, it appends a Reduction Cell at the end of the block.
    """
    current_input = input_id
    current_channels = in_channels
    
    # Add B Normal Cells
    for _ in range(num_normal_cells):
        g, current_input, node_counter = add_normal_cell(
            g, node_counter, current_input, current_channels
        )
    
    # Add 1 Reduction Cell
    if is_reduction:
        out_channels = current_channels * 2
        g, current_input, node_counter = add_reduction_cell(
            g, node_counter, current_input, current_channels, out_channels
        )
    else:
        out_channels = current_channels

    return g, current_input, node_counter, out_channels
    '''
