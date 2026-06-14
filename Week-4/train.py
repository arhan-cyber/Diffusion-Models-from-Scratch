import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import UNet
from diffusion import Diffusion


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def get_dataloader(
    batch_size=128
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,),
                (0.5,)
            )
        ]
    )

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )


def train(
    epochs=10,
    batch_size=128,
    learning_rate=1e-4,
    noise_steps=1000
):
    device = get_device()

    print(f"Using device: {device}")

    dataloader = get_dataloader(
        batch_size=batch_size
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        final_activation=None
    ).to(device)

    diffusion = Diffusion(
        noise_steps=noise_steps,
        device=device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    model.train()

    for epoch in range(epochs):

        epoch_loss = 0.0

        for images, _ in dataloader:

            images = images.to(device)

            t = diffusion.sample_timesteps(
                images.shape[0]
            )

            x_t, noise = diffusion.noise_images(
                images,
                t
            )

            predicted_noise = model(
                x_t,
                t
            )

            loss = F.mse_loss(
                predicted_noise,
                noise
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        average_loss = (
            epoch_loss
            / len(dataloader)
        )

        print(
            f"Epoch "
            f"[{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.6f}"
        )


def main():
    train()


if __name__ == "__main__":
    main()