import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import timeit
import copy
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Adjust paths to import your files
from morphisms.distill import train_student_with_distillation as orig_distill
from morphisms.distill_optimized import train_student_with_distillation as opt_distill

# --- Dummy Setup ---
class DummyModel(nn.Module):
    def __init__(self, out_features=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )
    def forward(self, x):
        return self.net(x)

def run_verification():
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Running test on: {device}")

    # 1. Create Fake Data
    torch.manual_seed(42)
    x = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16) # 4 batches per epoch

    # 2. Setup Models
    parent = DummyModel(out_features=10) # Matches labels
    child_orig = DummyModel(out_features=15) # Force shape mismatch to trigger your fallback logic
    
    # Deepcopy child to ensure identical starting weights
    child_opt = copy.deepcopy(child_orig)

    # 3. Train Original
    print("Training Original...")
    torch.manual_seed(42) # Ensure random dropouts/etc match if they existed
    orig_distill(
        parent_model=parent, child_model=child_orig, 
        dataloader=dataloader, device=device, epochs=2
    )

    # 4. Train Optimized
    print("Training Optimized...")
    torch.manual_seed(42)
    opt_distill(
        parent_model=parent, child_model=child_opt, 
        dataloader=dataloader, device=device, epochs=2
    )

    # 5. Assert Logical Equivalence
    is_equivalent = True
    for p_orig, p_opt in zip(child_orig.parameters(), child_opt.parameters()):
        if not torch.allclose(p_orig, p_opt, atol=1e-6):
            is_equivalent = False
            break

    print(f"\n✅ Weights Strictly Match: {is_equivalent}")
    assert is_equivalent, "Distillation logic failed! Final child weights differ."

    # 6. Benchmark
    print("\n--- Benchmarking Distillation (10 epochs) ---")
    
    # Re-instantiate fresh models for fair timing
    c1 = copy.deepcopy(child_orig)
    c2 = copy.deepcopy(child_orig)

    time_orig = timeit.timeit(lambda: orig_distill(parent, c1, dataloader, device, epochs=10), number=1)
    time_opt = timeit.timeit(lambda: opt_distill(parent, c2, dataloader, device, epochs=10), number=1)

    print(f"Original Time:  {time_orig:.4f} seconds")
    print(f"Optimized Time: {time_opt:.4f} seconds")
    print(f"Speedup:        {time_orig / time_opt:.2f}x faster")

if __name__ == "__main__":
    # Uncomment when your real imports are ready
    run_verification()
    pass



#1x faster no fayda
