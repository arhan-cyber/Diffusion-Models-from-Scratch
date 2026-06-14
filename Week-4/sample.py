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
    Load a trained DDPM UNet checkpoint.
    """

    model = UNet(
        in_channels=1,
        out_channels=1,
        final_activation=None
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Support both:
    # 1. Full training checkpoints
    # 2. Raw model.state_dict() checkpoints

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

    device = get_device()

    print(
        f"Using device: {device}"
    )

    checkpoint_path = (
        Path("checkpoints")
        / "ddpm_mnist.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    model = load_model(
        checkpoint_path,
        device
    )

    diffusion = Diffusion(
        device=device
    )

    print(
        "Generating samples..."
    )

    samples = diffusion.sample(
        model=model,
        n=64,
        image_size=28,
        channels=1
    )

    results_dir = Path(
        "results"
    )

    results_dir.mkdir(
        exist_ok=True
    )

    output_path = (
        results_dir
        / "samples_final.png"
    )

    save_image(
        samples,
        output_path,
        nrow=8
    )

    print(
        f"Saved samples to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()