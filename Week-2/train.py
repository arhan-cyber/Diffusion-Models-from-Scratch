import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from model import UNet
from dataset import NoisyDataset


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

def save_loss_curve(losses):

    plt.figure(figsize=(8, 5))

    plt.plot(losses)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.grid(True)

    save_path = os.path.join(
        RESULTS_DIR,
        "loss_curve.png"
    )

    plt.savefig(save_path)
    plt.close()

    print(f"Saved loss curve to {save_path}")


# ============================================================
# Training
# ============================================================

def train():

    transform = transforms.ToTensor()

    mnist_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_dataset = NoisyDataset(
        mnist_train,
        noise_min=NOISE_MIN,
        noise_max=NOISE_MAX
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
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

    epoch_losses = []

    print(f"Training on {DEVICE}")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0

        for noisy_imgs, clean_imgs in train_loader:

            noisy_imgs = noisy_imgs.to(DEVICE)
            clean_imgs = clean_imgs.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(noisy_imgs)

            loss = criterion(
                outputs,
                clean_imgs
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        epoch_losses.append(avg_loss)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Loss: {avg_loss:.6f}"
        )

        if epoch % SAVE_EVERY == 0:
            save_samples(
                model,
                train_loader,
                epoch,
                DEVICE
            )

    save_loss_curve(epoch_losses)

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