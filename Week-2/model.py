import torch
import torch.nn as nn

from blocks import DoubleConv, Down, Up

class UNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        depth=4
    ):
        super().__init__()

        self.depth = depth

        channels = [
            base_channels * (2 ** i)
            for i in range(depth + 1)
        ]

        self.inc = DoubleConv(
            in_channels,
            channels[0]
        )

        self.downs = nn.ModuleList()

        for i in range(depth):
            self.downs.append(
                Down(channels[i], channels[i+1])
            )

        self.ups = nn.ModuleList()

        for i in reversed(range(depth)):
            self.ups.append(
                Up(
                    channels[i+1] + channels[i],
                    channels[i]
                )
            )

        self.outc = nn.Conv2d(
            channels[0],
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        skips = []

        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        skips = skips[:-1][::-1]

        for up, skip in zip(self.ups, skips):
            x = up(x, skip)

        return self.outc(x)