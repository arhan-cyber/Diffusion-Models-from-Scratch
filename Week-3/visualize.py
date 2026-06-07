"""
Visualization of the forward diffusion process (noising trajectory).

Shows how a CIFAR-10 image progressively gets corrupted with noise
across the forward diffusion timesteps. Compares linear vs cosine schedules.

Output: results/noising_trajectory.png
  - 2 rows (linear, cosine schedules)
  - 6 columns (timesteps: 0, 100, 250, 500, 750, 999)
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets, transforms

from scheduler import NoiseScheduler


def load_cifar10_image(index: int = 0) -> torch.Tensor:
    """
    Load a CIFAR-10 image and normalize to [-1, 1].
    
    Args:
        index: Index of image in CIFAR-10 training set
        
    Returns:
        Normalized image tensor [3, 32, 32] in [-1, 1] range
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Download CIFAR-10 if not present
    cifar10 = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    
    image, _ = cifar10[index]
    
    # Normalize to [-1, 1]
    # ToTensor() gives [0, 1], so map to [-1, 1]
    image = image * 2.0 - 1.0
    
    return image


def denormalize_for_display(image: torch.Tensor) -> torch.Tensor:
    """
    Convert image from [-1, 1] to [0, 1] for display.
    
    Args:
        image: Tensor in [-1, 1]
        
    Returns:
        Tensor in [0, 1] clipped for display
    """
    return torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)


def visualize_noising_trajectory():
    """
    Create and save visualization of noising trajectories.
    
    Generates a 2×6 subplot figure:
      - Row 1: Linear schedule
      - Row 2: Cosine schedule
      - Columns: Timesteps [0, 100, 250, 500, 750, 999]
    """
    
    print("Loading CIFAR-10 image...")
    image = load_cifar10_image(index=0)  # Shape: [3, 32, 32]
    batch = image.unsqueeze(0)  # Shape: [1, 3, 32, 32]
    
    # Timesteps to visualize
    timesteps_to_show = [0, 100, 250, 500, 750, 999]
    num_timesteps = len(timesteps_to_show)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = batch.to(device)
    print(f"Device: {device}")
    
    # Create schedulers
    print("Initializing schedulers...")
    scheduler_linear = NoiseScheduler(
        timesteps=1000,
        schedule="linear",
        device=device,
    )
    
    scheduler_cosine = NoiseScheduler(
        timesteps=1000,
        schedule="cosine",
        device=device,
    )
    
    # Create figure with 2 rows, 6 columns
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(2, num_timesteps, figure=fig, hspace=0.3, wspace=0.2)
    
    schedules = [
        ("Linear Schedule", scheduler_linear),
        ("Cosine Schedule", scheduler_cosine),
    ]
    
    print("\nGenerating noised images...")
    for row_idx, (schedule_name, scheduler) in enumerate(schedules):
        print(f"  {schedule_name}...", end="")
        
        for col_idx, t_value in enumerate(timesteps_to_show):
            # Create timestep tensor
            t = torch.tensor([t_value], device=device)
            
            # Add noise
            xt, _ = scheduler.add_noise(batch, t)
            
            # Convert to numpy for display
            xt_display = denormalize_for_display(xt.squeeze(0).cpu())
            xt_np = xt_display.permute(1, 2, 0).numpy()
            
            # Create subplot
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(xt_np)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Title and labels
            if row_idx == 0:
                ax.set_title(f"t = {t_value}", fontsize=12, fontweight="bold")
            
            if col_idx == 0:
                ax.set_ylabel(schedule_name, fontsize=12, fontweight="bold")
        
        print(" Done")
    
    # Overall title
    fig.suptitle(
        "Forward Diffusion Process: CIFAR-10 Image Noising Trajectory\n"
        "Linear vs Cosine Schedules",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    
    # Create results directory if needed
    os.makedirs("results", exist_ok=True)
    
    # Save figure
    output_path = "results/noising_trajectory.png"
    print(f"\nSaving figure to {output_path}...")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print("Done!")
    
    plt.close(fig)


if __name__ == "__main__":
    visualize_noising_trajectory()
