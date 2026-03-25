import torch
import torch.nn as nn
import timeit
import sys
import os
# Add the root directory to the Python path so it can find the 'architectures' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from architectures.compiler import CompiledModel as OriginalCompiledModel
# (Make sure to import the optimized one as well once you save it)
from architectures.compiler_optimized import CompiledModel as OptimizedCompiledModel

# --- Mocks to simulate your architectures/graph.py environment ---
class MockNode:
    def __init__(self, op_type, params=None, parents=None):
        self.op_type = op_type
        self.params = params or {}
        self.parents = parents or []

class MockGraph:
    def __init__(self, nodes, output_node):
        self.nodes = nodes
        self.output_node = output_node
    
    def topological_sort(self):
        # A hardcoded valid sort for our test graph
        return [1, 2, 3, 4, 5, 6, 7]

def build_test_graph():
    """ Creates a branching network: Conv -> (Conv, Conv) -> Add -> Flatten -> Linear """
    nodes = {
        1: MockNode('identity', parents=[]),                             # Input
        2: MockNode('conv', {'in_channels': 3, 'out_channels': 16}, [1]),# Stem
        3: MockNode('conv', {'in_channels': 16, 'out_channels': 16}, [2]),# Branch A
        4: MockNode('conv', {'in_channels': 16, 'out_channels': 16}, [2]),# Branch B
        5: MockNode('add', parents=[3, 4]),                              # Merge
        6: MockNode('flatten', parents=[5]),                             # Flatten
        7: MockNode('linear', {'in_features': 16*32*32, 'out_features': 10}, [6]) # FC
    }
    return MockGraph(nodes, output_node=7)

# --- Test Suite ---
def run_verification():
    # 1. Setup
    torch.manual_seed(42)
    graph1 = build_test_graph()
    
    # Instantiate models (replace with actual imports)
    model_original = OriginalCompiledModel(graph1)
    
    torch.manual_seed(42) # Ensure identical weight initialization
    model_optimized = OptimizedCompiledModel(graph1)

    # Force identical weights just to be absolutely certain
    model_optimized.load_state_dict(model_original.state_dict())
    
    model_original.eval()
    model_optimized.eval()

    test_input = torch.randn(4, 3, 32, 32)

    # 2. Assert Logical Equivalence
    with torch.no_grad():
        out_orig = model_original(test_input)
        out_opt = model_optimized(test_input)
    
    is_equivalent = torch.allclose(out_orig, out_opt, atol=1e-6)
    print(f"✅ Outputs Strictly Match: {is_equivalent}")
    assert is_equivalent, "Logical equivalence failed! Outputs differ."

    # 3. Benchmark Performance
    print("\n--- Benchmarking Forward Pass (1000 runs) ---")
    
    def benchmark(model):
        with torch.no_grad():
            model(test_input)

    time_orig = timeit.timeit(lambda: benchmark(model_original), number=1000)
    time_opt = timeit.timeit(lambda: benchmark(model_optimized), number=1000)
    
    print(f"Original Time:  {time_orig:.4f} seconds")
    print(f"Optimized Time: {time_opt:.4f} seconds")
    print(f"Speedup:        {time_orig / time_opt:.2f}x faster")

if __name__ == "__main__":
    # Remove these dummy classes and plug in your actual models to test
    run_verification()
    pass