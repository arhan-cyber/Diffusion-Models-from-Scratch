import torch

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from diffusion import Diffusion


def test_prepare_noise_schedule():

    diffusion = Diffusion()

    beta = diffusion.beta

    assert beta.shape == (1000,)

    assert torch.isclose(
        beta[0],
        torch.tensor(1e-4)
    )

    assert torch.isclose(
        beta[-1],
        torch.tensor(0.02)
    )


def test_sample_timesteps():

    diffusion = Diffusion()

    t = diffusion.sample_timesteps(
        batch_size=32
    )

    assert t.shape == (32,)

    assert torch.all(t >= 1)

    assert torch.all(
        t < diffusion.noise_steps
    )

def test_noise_images_shape():

    diffusion = Diffusion()

    x = torch.randn(
        8,
        1,
        28,
        28
    )

    t = diffusion.sample_timesteps(
        x.shape[0]
    )

    x_t, noise = diffusion.noise_images(
        x,
        t
    )

    assert x_t.shape == x.shape

    assert noise.shape == x.shape

def test_noise_images_changes_input():

    diffusion = Diffusion()

    x = torch.randn(
        4,
        1,
        28,
        28
    )

    t = diffusion.sample_timesteps(
        x.shape[0]
    )

    x_t, _ = diffusion.noise_images(
        x,
        t
    )

    assert not torch.allclose(
        x,
        x_t
    )

def test_alpha_hat_monotonic():

    diffusion = Diffusion()

    alpha_hat = diffusion.alpha_hat

    assert torch.all(
        alpha_hat[:-1] >= alpha_hat[1:]
    )