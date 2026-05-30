import torch
import torch.nn as nn

from blocks import DoubleConv, Down, Up


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        depth=4,
        use_skip_connections=True,
        final_activation=None
    ):
        super().__init__()

        self.depth = depth
        self.use_skip_connections = use_skip_connections
        self.final_activation = final_activation

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
                Down(
                    channels[i],
                    channels[i + 1]
                )
            )

        self.ups = nn.ModuleList()

        for i in reversed(range(depth)):

            if use_skip_connections:
                up_in_channels = (
                    channels[i + 1] + channels[i]
                )
            else:
                up_in_channels = channels[i + 1]

            self.ups.append(
                Up(
                    up_in_channels,
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

        if self.use_skip_connections:

            for up, skip in zip(self.ups, skips):
                x = up(x, skip)

        else:

            for up in self.ups:
                x = up(x, None)

        x = self.outc(x)

        if self.final_activation == "sigmoid":
            x = torch.sigmoid(x)

        elif self.final_activation == "tanh":
            x = torch.tanh(x)

        return x

    def get_config(self):
        return {
            "depth": self.depth,
            "use_skip_connections": self.use_skip_connections,
            "final_activation": self.final_activation
        }