import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from model import UNet
from dataset import NoisyDataset
from tqdm import tqdm

from torch.cuda.amp import (
    autocast,
    GradScaler
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50

BASE_CHANNELS = 32
DEPTH = 4

NOISE_MIN = 0.05
NOISE_MAX = 0.50

SAVE_EVERY = 10

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(
    RESULTS_DIR,
    "checkpoints"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# Visualization
# ============================================================

def save_samples(model, dataloader, epoch, device):
    model.eval()

    noisy_imgs, clean_imgs = next(iter(dataloader))

    noisy_imgs = noisy_imgs.to(device)
    clean_imgs = clean_imgs.to(device)

    with torch.no_grad():
        outputs = model(noisy_imgs)

    noisy_imgs = noisy_imgs.cpu()
    clean_imgs = clean_imgs.cpu()
    outputs = outputs.cpu()

    n_samples = min(5, noisy_imgs.size(0))

    fig, axes = plt.subplots(n_samples, 3, figsize=(8, 2 * n_samples))

    for i in range(n_samples):

        axes[i, 0].imshow(
            noisy_imgs[i].squeeze(),
            cmap="gray"
        )
        axes[i, 0].set_title("Noisy")

        axes[i, 1].imshow(
            outputs[i].squeeze(),
            cmap="gray"
        )
        axes[i, 1].set_title("Denoised")

        axes[i, 2].imshow(
            clean_imgs[i].squeeze(),
            cmap="gray"
        )
        axes[i, 2].set_title("Ground Truth")

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()

    save_path = os.path.join(
        RESULTS_DIR,
        f"samples_epoch_{epoch}.png"
    )

    plt.savefig(save_path)
    plt.close()

    print(f"Saved samples to {save_path}")


# ============================================================
# Loss Curve
# ============================================================
def save_loss_curve(
    train_losses,
    val_losses
):
    plt.figure(figsize=(8, 5))

    plt.plot(
        train_losses,
        label="Train"
    )

    plt.plot(
        val_losses,
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)

    save_path = os.path.join(
        RESULTS_DIR,
        "loss_curve.png"
    )

    plt.savefig(save_path)
    plt.close()


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for noisy_imgs, clean_imgs in dataloader:

            noisy_imgs = noisy_imgs.to(
                device,
                non_blocking=True
            )

            clean_imgs = clean_imgs.to(
                device,
                non_blocking=True
            )

            with autocast(
                enabled=torch.cuda.is_available()
            ):
                outputs = model(noisy_imgs)

                loss = criterion(
                    outputs,
                    clean_imgs
                )

            running_loss += loss.item()

    return running_loss / len(dataloader)


# ============================================================
# Training
# ============================================================

def train():

    transform = transforms.ToTensor()

    full_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_base, val_base = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )

    train_dataset = NoisyDataset(
        train_base,
        noise_min=NOISE_MIN,
        noise_max=NOISE_MAX
    )

    val_dataset = NoisyDataset(
        val_base,
        noise_min=NOISE_MIN,
        noise_max=NOISE_MAX
    )

    train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True
    
    )

    val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=BASE_CHANNELS,
        depth=DEPTH
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scaler = GradScaler(
    enabled=torch.cuda.is_available()
    )

    train_losses = []
    val_losses = []

    print(f"Training on {DEVICE}")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            leave=False
        )

        for noisy_imgs, clean_imgs in progress_bar:

            noisy_imgs = noisy_imgs.to(DEVICE, non_blocking=True)
            clean_imgs = clean_imgs.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(
                enabled=torch.cuda.is_available()
            ):
                outputs = model(noisy_imgs)

                loss = criterion(
                    outputs,
                    clean_imgs
                )

            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )


            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(
                loss=f"{loss.item():.6f}"
            )

        avg_loss = running_loss / len(train_loader)
        val_loss = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"epoch_{epoch:03d}.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "train_loss":
                    avg_loss,
                "val_loss":
                    val_loss
            },
            checkpoint_path
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_model_path = os.path.join(
                RESULTS_DIR,
                "best_model.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "train_loss":
                        avg_loss,
                    "val_loss":
                        val_loss
                },
                best_model_path
            )

            print(
                f"New best model saved "
                f"(val_loss={val_loss:.6f})"
            )

        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {avg_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        if epoch % SAVE_EVERY == 0:
            save_samples(
                model,
                train_loader,
                epoch,
                DEVICE
            )

    save_loss_curve(train_losses, val_losses)

    model_path = os.path.join(
        RESULTS_DIR,
        "unet_denoiser.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(f"Model saved to {model_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train()