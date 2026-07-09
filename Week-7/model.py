import torch
import torch.nn as nn
from blocks import DoubleConv, Down, Up
from embeddings import TimeEmbedding

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        depth=3,
        emb_dim=256,
        use_skip_connections=True,
        use_batchnorm=True,
        final_activation=None,
        num_classes=None
    ):
        super().__init__()
        self.time_embedding = TimeEmbedding(emb_dim=emb_dim)
        self.depth = depth
        self.use_skip_connections = use_skip_connections
        self.use_batchnorm = use_batchnorm
        self.final_activation = final_activation
        self.num_classes = num_classes

        self._in_channels = in_channels
        self._base_channels = base_channels

        # Channel progression
        channels = [
            base_channels * (2 ** i)
            for i in range(depth + 1)
        ]

        # Initial convolution block
        self.inc = DoubleConv(
            in_channels=in_channels,
            out_channels=channels[0],
            emb_dim=emb_dim,
            use_batchnorm=use_batchnorm
        )

        # Encoder
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(
                Down(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    emb_dim=emb_dim,
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
                    emb_dim=emb_dim,
                    use_skip_connections=use_skip_connections,
                    use_batchnorm=use_batchnorm
                )
            )

        # Class embedding if conditional
        if num_classes is not None:
            # We use num_classes + 1 to account for the null class (index num_classes)
            self.class_embedding = nn.Embedding(num_classes + 1, emb_dim)

        self.emb_dim = emb_dim
        self._emb_dim = emb_dim

        # Final projection
        self.outc = nn.Conv2d(
            channels[0],
            out_channels,
            kernel_size=1
        )

    def forward(self, x, t, c=None):
        emb = self.time_embedding(t)

        if self.num_classes is not None:
            if c is None:
                # Default to the null class index (self.num_classes)
                c = torch.full((x.shape[0],), self.num_classes, dtype=torch.long, device=x.device)
            c_emb = self.class_embedding(c)
            emb = emb + c_emb

        skips = []
        x = self.inc(x, emb)
        skips.append(x)

        for down in self.downs:
            x = down(x, emb)
            skips.append(x)

        skips = skips[:-1][::-1]

        for up, skip in zip(self.ups, skips):
            x = up(
                x,
                skip if self.use_skip_connections else None,
                emb
            )

        x = self.outc(x)

        if self.final_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.final_activation == "tanh":
            x = torch.tanh(x)

        return x

    def get_config(self):
        return {
            "in_channels": self._in_channels,
            "out_channels": self.outc.out_channels,
            "base_channels": self._base_channels,
            "depth": self.depth,
            "use_skip_connections": self.use_skip_connections,
            "use_batchnorm": self.use_batchnorm,
            "final_activation": self.final_activation,
            "emb_dim": self._emb_dim,
            "num_classes": self.num_classes
        }

    @property
    def num_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
