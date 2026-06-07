"""
Compare SNR (Signal-to-Noise Ratio) curves for linear vs cosine schedules.

SNR(t) = α̅(t) / (1 - α̅(t))

The cosine schedule maintains higher SNR at mid-timesteps, leading to
smoother noise injection and better sample quality during training.

Output: results/linear_vs_cosine.png
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from scheduler import NoiseScheduler


def plot_snr_comparison():
    """
    Create and save SNR comparison plot.
    
    Features:
      - Log-scale SNR (varies over many orders of magnitude)
      - Both linear and cosine schedules on same plot
      - Key timesteps marked (0, 100, 250, 500, 750, 999)
      - Grid for readability
    """
    
    print("Initializing schedulers...")
    scheduler_linear = NoiseScheduler(
        timesteps=1000,
        schedule="linear",
        device="cpu",
    )
    
    scheduler_cosine = NoiseScheduler(
        timesteps=1000,
        schedule="cosine",
        device="cpu",
    )
    
    print("Computing SNR curves...")
    snr_linear = scheduler_linear.get_snr().cpu().numpy()
    snr_cosine = scheduler_cosine.get_snr().cpu().numpy()
    
    timesteps = np.arange(1000)
    
    # Key timesteps for visualization
    key_timesteps = [0, 100, 250, 500, 750, 999]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot SNR curves with log scale
    ax.semilogy(timesteps, snr_linear, label="Linear Schedule", linewidth=2.5, color="C0")
    ax.semilogy(timesteps, snr_cosine, label="Cosine Schedule", linewidth=2.5, color="C1")
    
    # Mark key timesteps
    snr_linear_at_key = [snr_linear[t] for t in key_timesteps]
    snr_cosine_at_key = [snr_cosine[t] for t in key_timesteps]
    
    ax.scatter(
        key_timesteps,
        snr_linear_at_key,
        s=100,
        marker="o",
        color="C0",
        zorder=5,
        edgecolors="black",
        linewidths=1,
    )
    
    ax.scatter(
        key_timesteps,
        snr_cosine_at_key,
        s=100,
        marker="s",
        color="C1",
        zorder=5,
        edgecolors="black",
        linewidths=1,
    )
    
    # Add vertical lines at key timesteps for reference
    for t in key_timesteps:
        ax.axvline(x=t, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    
    # Labels and formatting
    ax.set_xlabel("Timestep (t)", fontsize=12, fontweight="bold")
    ax.set_ylabel("SNR(t) [log scale]", fontsize=12, fontweight="bold")
    ax.set_title(
        "Signal-to-Noise Ratio: Linear vs Cosine Schedules\n"
        "SNR(t) = α̅(t) / (1 - α̅(t))",
        fontsize=13,
        fontweight="bold",
    )
    
    ax.grid(True, which="both", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(fontsize=11, loc="upper right")
    
    # X-axis limits
    ax.set_xlim(0, 1000)
    
    # Create results directory if needed
    os.makedirs("results", exist_ok=True)
    
    # Save figure
    output_path = "results/linear_vs_cosine.png"
    print(f"\nSaving figure to {output_path}...")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print("Done!")
    
    plt.close(fig)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SNR Summary Statistics")
    print("="*60)
    print(f"{'Timestep':<12} {'Linear SNR':<15} {'Cosine SNR':<15} {'Ratio':<10}")
    print("-"*60)
    
    for t in key_timesteps:
        linear_snr = snr_linear[t]
        cosine_snr = snr_cosine[t]
        ratio = cosine_snr / (linear_snr + 1e-8)
        print(f"{t:<12} {linear_snr:<15.4f} {cosine_snr:<15.4f} {ratio:<10.2f}x")
    
    print("="*60)
    print(f"\nKey insight: Cosine schedule maintains higher SNR at mid-")
    print(f"timesteps, leading to smoother noise injection and better")
    print(f"sample quality during training.")
    print("="*60)


if __name__ == "__main__":
    plot_snr_comparison()
