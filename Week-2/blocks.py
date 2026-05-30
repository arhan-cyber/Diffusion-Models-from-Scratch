import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_batchnorm=True
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm
            )
        ]

        if use_batchnorm:
            layers.append(
                nn.BatchNorm2d(out_channels)
            )

        layers.append(
            nn.ReLU(inplace=True)
        )

        layers.append(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm
            )
        )

        if use_batchnorm:
            layers.append(
                nn.BatchNorm2d(out_channels)
            )

        layers.append(
            nn.ReLU(inplace=True)
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_batchnorm=True
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                use_batchnorm=use_batchnorm
            )
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(
        self,
        decoder_channels,
        skip_channels,
        out_channels,
        use_skip_connections=True,
        use_batchnorm=True,
        bilinear=True
    ):
        super().__init__()

        self.use_skip_connections = use_skip_connections

        if bilinear:

            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True
            )

        else:
            raise NotImplementedError(
                "ConvTranspose2d mode is not currently "
                "supported in this implementation."
            )

        conv_in_channels = decoder_channels

        if use_skip_connections:
            conv_in_channels += skip_channels

        self.conv = DoubleConv(
            in_channels=conv_in_channels,
            out_channels=out_channels,
            use_batchnorm=use_batchnorm
        )

    def forward(
        self,
        x1,
        x2=None
    ):
        x1 = self.up(x1)

        if self.use_skip_connections:

            if x2 is None:
                raise ValueError(
                    "Skip connections enabled but "
                    "skip tensor is None."
                )

            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)

            x1 = F.pad(
                x1,
                [
                    diffX // 2,
                    diffX - diffX // 2,
                    diffY // 2,
                    diffY - diffY // 2
                ]
            )

            x1 = torch.cat(
                [x2, x1],
                dim=1
            )

        return self.conv(x1)