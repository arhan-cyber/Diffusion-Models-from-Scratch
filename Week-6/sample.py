import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image

from diffusion import Diffusion
from model import UNet


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_model(
    checkpoint_path,
    device
):
    """
    Load a trained conditional UNet checkpoint.
    """
    # Initialize the UNet with 10 classes
    model = UNet(
        in_channels=1,
        out_channels=1,
        final_activation=None,
        num_classes=10
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Support both full training checkpoints and raw state_dict checkpoints
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Sample from Class-Conditional DDPM with CFG")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ddpm_mnist.pt", help="Path to checkpoint")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Please run train.py first to train a model."
        )

    model = load_model(
        checkpoint_path,
        device
    )

    diffusion = Diffusion(
        device=device
    )

    results_dir = Path("results")
    results_dir.mkdir(
        exist_ok=True
    )

    # 1. Generate Conditional Grid (Same noise vector -> different classes)
    print("Generating conditional grid (same noise, different classes)...")
    num_noise_vectors = 8
    # Shared base noise
    z_base = torch.randn(
        num_noise_vectors,
        1,
        28,
        28,
        device=device
    )
    
    # 10 rows (digits 0-9), each using the same 8 noise vectors
    cond_samples = []
    for digit in range(10):
        labels = torch.full((num_noise_vectors,), digit, dtype=torch.long, device=device)
        samples = diffusion.sample(
            model=model,
            n=num_noise_vectors,
            image_size=28,
            channels=1,
            labels=labels,
            cfg_scale=3.0,
            x_start=z_base
        )
        cond_samples.append(samples)
    
    cond_grid = torch.cat(cond_samples, dim=0)
    cond_grid_path = results_dir / "conditional_grid.png"
    save_image(cond_grid, cond_grid_path, nrow=num_noise_vectors)
    print(f"Saved conditional grid to: {cond_grid_path}")

    # 2. Generate Guidance Scale Sweep (Same class with w = 1, 3, 5, 10)
    print("Generating guidance scale sweep (w = 1, 3, 5, 10)...")
    target_digit = 7
    w_values = [1.0, 3.0, 5.0, 10.0]
    sweep_samples = []
    
    for w in w_values:
        labels = torch.full((num_noise_vectors,), target_digit, dtype=torch.long, device=device)
        samples = diffusion.sample(
            model=model,
            n=num_noise_vectors,
            image_size=28,
            channels=1,
            labels=labels,
            cfg_scale=w,
            x_start=z_base
        )
        sweep_samples.append(samples)
        
    sweep_grid = torch.cat(sweep_samples, dim=0)
    sweep_grid_path = results_dir / "guidance_scale_sweep.png"
    save_image(sweep_grid, sweep_grid_path, nrow=num_noise_vectors)
    print(f"Saved guidance scale sweep to: {sweep_grid_path}")

    # 3. Generate Unconditional vs CFG comparison
    print("Generating unconditional vs CFG comparison...")
    # Row 1: Unconditional (using the null label 10)
    uncond_labels = torch.full((num_noise_vectors,), 10, dtype=torch.long, device=device)
    uncond_samples = diffusion.sample(
        model=model,
        n=num_noise_vectors,
        image_size=28,
        channels=1,
        labels=uncond_labels,
        cfg_scale=0.0,
        x_start=z_base
    )
    
    # Row 2: CFG sample for a target digit (e.g. 3) with w = 3.0
    cfg_labels = torch.full((num_noise_vectors,), 3, dtype=torch.long, device=device)
    cfg_samples = diffusion.sample(
        model=model,
        n=num_noise_vectors,
        image_size=28,
        channels=1,
        labels=cfg_labels,
        cfg_scale=3.0,
        x_start=z_base
    )
    
    comparison_grid = torch.cat([uncond_samples, cfg_samples], dim=0)
    comparison_path = results_dir / "unconditional_vs_cfg.png"
    save_image(comparison_grid, comparison_path, nrow=num_noise_vectors)
    print(f"Saved unconditional vs CFG comparison to: {comparison_path}")


if __name__ == "__main__":
    main()
