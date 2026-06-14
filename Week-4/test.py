import torch

from model import UNet


def main():

    model = UNet()

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

    y = model(x, t)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Contains NaN:", torch.isnan(y).any().item())


if __name__ == "__main__":
    main()