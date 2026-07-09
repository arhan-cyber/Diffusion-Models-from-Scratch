import math
import torch
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(
                half_dim,
                device=device
            ) * -embeddings
        )

        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat(
            (
                embeddings.sin(),
                embeddings.cos()
            ),
            dim=-1
        )

        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(
                embeddings,
                (0, 1)
            )

        return embeddings


class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.sinusoidal_embedding = SinusoidalPositionEmbeddings(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, t):
        t = self.sinusoidal_embedding(t)
        return self.mlp(t)
