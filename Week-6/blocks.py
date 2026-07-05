import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_dim,
        use_batchnorm=True
    ):
        super().__init__()

        # Unpacking into distinct layers to allow embedding injection
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batchnorm
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)

        # Time embedding projection layer
        self.emb_layer = nn.Sequential(
            nn.SiLU(),  # Standard non-linearity for time embeddings in diffusion
            nn.Linear(emb_dim, out_channels)
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batchnorm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, t_emb):
        # First convolution pass
        h = self.conv1(x)
        h = self.bn1(h)

        # Project and broadcast the timestep embedding
        # From [batch_size, emb_dim] -> [batch_size, out_channels, 1, 1]
        emb = self.emb_layer(t_emb)
        emb = emb[:, :, None, None]

        # Inject embedding directly into the feature maps before activation
        h = h + emb
        h = self.relu1(h)

        # Second convolution pass
        h = self.conv2(h)
        h = self.bn2(h)
        return self.relu2(h)


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_dim,
        use_batchnorm=True
    ):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            emb_dim=emb_dim,
            use_batchnorm=use_batchnorm
        )

    def forward(self, x, t_emb):
        x = self.maxpool(x)
        return self.conv(x, t_emb)


class Up(nn.Module):
    def __init__(
        self,
        decoder_channels,
        skip_channels,
        out_channels,
        emb_dim,
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
                "ConvTranspose2d mode is not currently supported."
            )

        conv_in_channels = decoder_channels
        if use_skip_connections:
            conv_in_channels += skip_channels

        self.conv = DoubleConv(
            in_channels=conv_in_channels,
            out_channels=out_channels,
            emb_dim=emb_dim,
            use_batchnorm=use_batchnorm
        )

    def forward(self, x1, x2=None, t_emb=None):
        x1 = self.up(x1)

        if self.use_skip_connections:
            if x2 is None:
                raise ValueError("Skip connections enabled but skip tensor is None.")

            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)

            x1 = F.pad(
                x1,
                [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )

            x1 = torch.cat([x2, x1], dim=1)

        if t_emb is None:
            raise ValueError("Timestep embedding `t_emb` must be passed during forward pass.")

        return self.conv(x1, t_emb)
