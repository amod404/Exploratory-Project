import torch
import torch.nn as nn
import timeit
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Adjust paths to import your files
from morphisms.exact import initialize_conv_as_identity as orig_init_conv
from morphisms.exact_optimized import initialize_conv_as_identity as opt_init_conv

from morphisms.exact import inherit_weights_net2wider as orig_inherit
from morphisms.exact_optimized import inherit_weights_net2wider as opt_inherit

class DummyModel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleDict({
            '1': nn.Conv2d(3, channels, kernel_size=3, padding=1),
            '2': nn.BatchNorm2d(channels)
        })

def test_conv_identity():
    print("--- Testing Identity Init ---")
    conv_orig = nn.Conv2d(16, 16, 3, bias=False)
    conv_opt = nn.Conv2d(16, 16, 3, bias=False)
    
    # We don't check strict matching here because both initialize to 0s and 1s deterministically
    orig_init_conv(conv_orig)
    opt_init_conv(conv_opt)
    
    assert torch.allclose(conv_orig.weight, conv_opt.weight)
    print("✅ Identity logic verified.")

    # --- Benchmarking ---
    print("\nBenchmarking Identity Init (1000 runs)...")
    runs = 1000
    t_orig = timeit.timeit(lambda: orig_init_conv(conv_orig), number=runs)
    t_opt = timeit.timeit(lambda: opt_init_conv(conv_opt), number=runs)
    
    print(f"Original Time:  {t_orig:.4f} seconds")
    print(f"Optimized Time: {t_opt:.4f} seconds")
    print(f"Speedup:        {t_orig / t_opt:.2f}x faster")

def test_net2wider_inheritance():
    print("\n--- Testing Net2Wider Inheritance ---")
    torch.manual_seed(42)
    parent = DummyModel(16)
    
    child_orig = DummyModel(32)
    child_opt = DummyModel(32)
    
    # Force children to have identical starting random states to isolate the inherited weights
    child_opt.load_state_dict(child_orig.state_dict())

    torch.manual_seed(42) # Sync the random selection of filters
    orig_inherit(parent, child_orig, '1', widen_by=16)
    
    torch.manual_seed(42) # Sync the random selection of filters
    opt_inherit(parent, child_opt, '1', widen_by=16)

    is_equivalent = torch.allclose(child_orig.layers['1'].weight, child_opt.layers['1'].weight)
    print(f"✅ Weights Strictly Match: {is_equivalent}")

    # --- Benchmarking ---
    print("\nBenchmarking Net2Wider Inheritance (1000 runs)...")
    runs = 1000
    
    # We fix the seeds inside the lambda to ensure fair random sampling overhead
    def run_orig():
        torch.manual_seed(42)
        orig_inherit(parent, child_orig, '1', widen_by=16)
        
    def run_opt():
        torch.manual_seed(42)
        opt_inherit(parent, child_opt, '1', widen_by=16)

    t_orig = timeit.timeit(run_orig, number=runs)
    t_opt = timeit.timeit(run_opt, number=runs)

    print(f"Original Time:  {t_orig:.4f} seconds")
    print(f"Optimized Time: {t_opt:.4f} seconds")
    print(f"Speedup:        {t_orig / t_opt:.2f}x faster")

if __name__ == "__main__":
    test_conv_identity()
    test_net2wider_inheritance()