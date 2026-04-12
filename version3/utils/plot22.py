################################################################################
# FILE:   plot_interactive_3d.py
# DESCRIPTION: Standalone script to load history.pkl and render a rotatable 
#              3D surface plot of the Pareto front.
################################################################################

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_interactive_3d(history):
    """
    Renders an interactive 3D scatter and surface plot from the given history.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    generations = sorted(list(history.keys()))
    
    # Use a colormap to differentiate generations visually
    colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))
    has_data = False
    
    for idx, gen in enumerate(generations):
        pop = history[gen]
        
        # Safely extract objectives
        params = [ind.get('params') for ind in pop if ind.get('params') is not None]
        flops = [ind.get('flops') for ind in pop if ind.get('flops') is not None]
        val_error = [ind.get('val_error') for ind in pop if ind.get('val_error') is not None] 
        
        if len(params) == len(flops) == len(val_error) and len(params) > 0:
            has_data = True
            
            # 1. Scatter the exact models as solid dots
            ax.scatter(
                params, flops, val_error, 
                color=colors[idx], label=f"Gen {gen}", 
                alpha=1.0, s=60, edgecolor='w', linewidth=0.5
            )
            
            # 2. Draw the 3D Surface Mesh (Curvature) connecting the models
            if len(params) >= 3:
                try:
                    ax.plot_trisurf(
                        params, flops, val_error, 
                        color=colors[idx], alpha=0.3, edgecolor='none'
                    )
                except Exception as e:
                    print(f"Skipped surface mesh for Gen {gen} due to collinear points: {e}")
    
    if not has_data:
        print("ERROR: Cannot plot 3D surface. Missing one or more required objectives (params, flops, val_error).")
        plt.close()
        return

    # Formatting and labels
    ax.set_title("Interactive 3D Pareto Front Evolution", fontsize=16, pad=20)
    ax.set_xlabel("Parameters", fontsize=12, labelpad=10)
    ax.set_ylabel("FLOPs", fontsize=12, labelpad=10)
    ax.set_zlabel("Validation Error", fontsize=12, labelpad=10)
    
    # Initial camera angle (can be rotated interactively once opened)
    ax.view_init(elev=20, azim=-45)
    
    if len(generations) > 1:
        # Move legend outside the main graph area
        ax.legend(title="Generations", bbox_to_anchor=(1.15, 0.9), loc='upper left')
        
    print("Opening interactive 3D plot window...")
    print("-> CLICK AND DRAG to rotate.")
    print("-> SCROLL to zoom.")
    print("-> Close the window to terminate the script.")
    
    # This triggers the interactive window
    plt.show()

def main():
    # =========================================================================
    # CONFIGURATION: Set your specific timestamp / experiment folder here
    # =========================================================================
    RUN_DIR = r"results_CIFAR-10_20260412_012124" 
    # version2/results_CIFAR-10_20260411_231008
    # version2/results_CIFAR-10_20260412_012124
    
    history_path = os.path.join(RUN_DIR, "history", "history.pkl")
    
    if not os.path.exists(history_path):
        print(f"ERROR: Cannot find history file at {history_path}")
        print("Please check the RUN_DIR path and ensure the experiment saved a history.pkl file.")
        return
        
    print(f"Loading history from {history_path}...")
    with open(history_path, "rb") as f:
        history = pickle.load(f)
        
    print(f"Successfully loaded {len(history)} generations.")
    plot_interactive_3d(history)

if __name__ == "__main__":
    main()