import torch
import torch.nn as nn

from blocks import DoubleConv, Down, Up


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        depth=3,
        use_skip_connections=True,
        use_batchnorm=True,
        final_activation=None
    ):
        super().__init__()

        self.depth = depth
        self.use_skip_connections = use_skip_connections
        self.use_batchnorm = use_batchnorm
        self.final_activation = final_activation

        # Channel progression
        # Example:
        # base_channels=64, depth=4
        # [64, 128, 256, 512, 1024]
        channels = [
            base_channels * (2 ** i)
            for i in range(depth + 1)
        ]

        # Initial convolution block
        self.inc = DoubleConv(
            in_channels=in_channels,
            out_channels=channels[0],
            use_batchnorm=use_batchnorm
        )

        # Encoder
        self.downs = nn.ModuleList()

        for i in range(depth):
            self.downs.append(
                Down(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    use_batchnorm=use_batchnorm
                )
            )

        # Decoder
        self.ups = nn.ModuleList()

        for i in reversed(range(depth)):

            self.ups.append(
                Up(
                    decoder_channels=channels[i + 1],
                    skip_channels=channels[i],
                    out_channels=channels[i],
                    use_skip_connections=use_skip_connections,
                    use_batchnorm=use_batchnorm
                )
            )

        # Final projection
        self.outc = nn.Conv2d(
            channels[0],
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        skips = []

        # Encoder
        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Remove bottleneck feature map
        skips = skips[:-1][::-1]

        # Decoder
        for up, skip in zip(self.ups, skips):

            x = up(
                x,
                skip if self.use_skip_connections else None
            )

        # Output projection
        x = self.outc(x)

        # Optional final activation
        if self.final_activation == "sigmoid":
            x = torch.sigmoid(x)

        elif self.final_activation == "tanh":
            x = torch.tanh(x)

        return x

    def get_config(self):

        return {
            "in_channels": self.inc.block[0].in_channels,
            "out_channels": self.outc.out_channels,
            "base_channels": self.outc.in_channels,
            "depth": self.depth,
            "use_skip_connections": self.use_skip_connections,
            "use_batchnorm": self.use_batchnorm,
            "final_activation": self.final_activation
        }

    @property
    def num_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )