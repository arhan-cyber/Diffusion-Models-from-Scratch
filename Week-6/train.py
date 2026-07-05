import time
from torchvision.utils import save_image
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from model import UNet
from diffusion import Diffusion

def save_samples(
    model,
    diffusion,
    epoch,
    results_dir,
    num_samples=64,
    device="cpu"
):
    # For training progress, we can sample digits 0-7 (8 samples each) to fill an 8x8 grid
    labels = torch.arange(8, device=device).repeat_interleave(8)
    
    # We will pass cfg_scale (w) to the sample function. Let's use w=3.0 as a good default.
    samples = diffusion.sample(
        model=model,
        n=num_samples,
        image_size=28,
        channels=1,
        labels=labels,
        cfg_scale=3.0
    )

    output_path = (
        results_dir
        / f"samples_epoch_{epoch}.png"
    )

    save_image(
        samples,
        output_path,
        nrow=8
    )

    print(
        f"Saved samples: "
        f"{output_path}"
    )

def save_loss_curve(
    loss_history,
    output_path
):
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        loss_history
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Conditional DDPM Training Loss"
    )

    plt.tight_layout()

    plt.savefig(
        output_path
    )

    plt.close()


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def get_dataloader(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,),
                (0.5,)
            ),
        ]
    )

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def train(
    epochs=15,  # Increased standard training epochs if needed, or keeping it short
    batch_size=128,
    learning_rate=1e-4,
    noise_steps=1000,
    sample_every=2
):
    device = get_device()

    print(f"Using device: {device}")
    checkpoint_dir = Path(
        "checkpoints"
    )

    checkpoint_dir.mkdir(
        exist_ok=True
    )
    results_dir = Path(
        "results"
    )

    results_dir.mkdir(
        exist_ok=True
    )

    dataloader = get_dataloader(
        batch_size=batch_size
    )

    # UNet with 10 MNIST classes
    model = UNet(
        in_channels=1,
        out_channels=1,
        final_activation=None,
        num_classes=10
    ).to(device)

    diffusion = Diffusion(
        noise_steps=noise_steps,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()

    total_start = time.time()

    epoch_bar = tqdm(
        range(epochs),
        desc="Training",
        position=0,
    )
    best_loss = float("inf")
    loss_history = []

    for epoch in epoch_bar:
        epoch_start = time.time()
        epoch_loss = 0.0
        
        # Track statistics to verify the ~15% null label dropout rate
        total_dropped = 0
        total_labels = 0

        batch_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
            position=1,
        )

        for batch_idx, (images, labels) in enumerate(batch_bar):

            images = images.to(device)
            labels = labels.to(device)

            t = diffusion.sample_timesteps(
                images.shape[0]
            )

            x_t, noise = diffusion.noise_images(
                images,
                t,
            )

            # Random label dropout: replace label with 10 (null class) with ~15% probability
            dropout_mask = torch.rand(labels.shape[0], device=device) < 0.15
            cond_labels = torch.where(dropout_mask, torch.tensor(10, device=device), labels)
            
            total_dropped += dropout_mask.sum().item()
            total_labels += labels.shape[0]

            predicted_noise = model(
                x_t,
                t,
                cond_labels
            )

            loss = F.mse_loss(
                predicted_noise,
                noise,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            epoch_loss += loss_value

            avg_loss = (
                epoch_loss
                / (batch_idx + 1)
            )

            batch_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                avg=f"{avg_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        average_loss = (
            epoch_loss
            / len(dataloader)
        )
        loss_history.append(
            average_loss
        )
        
        dropped_percent = (total_dropped / total_labels) * 100
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.6f} "
            f"Null Label Dropout Rate: {dropped_percent:.1f}%"
        )
        
        if average_loss < best_loss:
            best_loss = average_loss
            checkpoint_path = (
                checkpoint_dir
                / "ddpm_mnist.pt"
            )

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": average_loss,
                },
                checkpoint_path
            )
            print(
                f"Saved checkpoint: "
                f"{checkpoint_path}"
            )

        epoch_time = (
            time.time()
            - epoch_start
        )

        epoch_bar.set_postfix(
            loss=f"{average_loss:.4f}",
            time=f"{epoch_time:.1f}s",
        )

        if (
            (epoch + 1)
            % sample_every
            == 0
        ):
            model.eval()
            with torch.no_grad():
                save_samples(
                    model=model,
                    diffusion=diffusion,
                    epoch=epoch + 1,
                    results_dir=results_dir,
                    device=device
                )
            model.train()

    total_time = (
        time.time()
        - total_start
    )

    print(
        f"\nTraining completed "
        f"in {total_time:.1f}s"
    )
    loss_curve_path = (
        results_dir
        / "loss_curve.png"
    )

    save_loss_curve(
        loss_history,
        loss_curve_path
    )

    print(
        f"Saved loss curve: "
        f"{loss_curve_path}"
    )
    
    model.eval()
    with torch.no_grad():
        save_samples(
            model=model,
            diffusion=diffusion,
            epoch="final",
            results_dir=results_dir,
            device=device
        )


def main():
    train()
    

if __name__ == "__main__":
    main()
