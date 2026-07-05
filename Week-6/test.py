import torch

from model import UNet


def main():

    # Initialize conditional UNet with 10 classes
    model = UNet(num_classes=10)

    x = torch.randn(
        4,
        1,
        28,
        28
    )

    t = torch.randint(
        0,
        1000,
        (4,)
    )

    # Class conditioning labels (0-9)
    c = torch.randint(
        0,
        10,
        (4,)
    )

    y = model(x, t, c)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Contains NaN:", torch.isnan(y).any().item())


if __name__ == "__main__":
    main()
