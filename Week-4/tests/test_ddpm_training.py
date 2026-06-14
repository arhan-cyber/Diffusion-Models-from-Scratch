import torch
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from diffusion import Diffusion
from model import UNet


def test_ddpm_training_step_runs():
    """
    Verify that a complete DDPM training step executes successfully.

    Covers:
    - timestep sampling
    - forward diffusion
    - UNet forward pass
    - MSE loss computation
    - backward pass
    """

    device = "cpu"

    model = UNet().to(device)

    diffusion = Diffusion(
        device=device
    )

    images = torch.randn(
        8,
        1,
        28,
        28,
        device=device
    )

    t = diffusion.sample_timesteps(
        images.shape[0]
    )

    x_t, noise = diffusion.noise_images(
        images,
        t
    )

    predicted_noise = model(
        x_t,
        t
    )

    loss = F.mse_loss(
        predicted_noise,
        noise
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_ddpm_output_shape_matches_input():
    """
    Verify that the UNet predicts noise with the same shape
    as the input image tensor.
    """

    device = "cpu"

    model = UNet().to(device)

    diffusion = Diffusion(
        device=device
    )

    images = torch.randn(
        8,
        1,
        28,
        28,
        device=device
    )

    t = diffusion.sample_timesteps(
        images.shape[0]
    )

    x_t, noise = diffusion.noise_images(
        images,
        t
    )

    predicted_noise = model(
        x_t,
        t
    )

    assert predicted_noise.shape == images.shape
    assert noise.shape == images.shape


def test_ddpm_backward_produces_gradients():
    """
    Verify that gradients are produced during backpropagation.
    """

    device = "cpu"

    model = UNet().to(device)

    diffusion = Diffusion(
        device=device
    )

    images = torch.randn(
        8,
        1,
        28,
        28,
        device=device
    )

    t = diffusion.sample_timesteps(
        images.shape[0]
    )

    x_t, noise = diffusion.noise_images(
        images,
        t
    )

    predicted_noise = model(
        x_t,
        t
    )

    loss = F.mse_loss(
        predicted_noise,
        noise
    )

    loss.backward()

    gradients_found = False

    for parameter in model.parameters():

        if (
            parameter.requires_grad
            and parameter.grad is not None
        ):
            gradients_found = True
            break

    assert gradients_found


def test_ddpm_parameters_receive_nonzero_gradients():
    """
    Verify that gradients are not only present but non-zero.
    """

    device = "cpu"

    model = UNet().to(device)

    diffusion = Diffusion(
        device=device
    )

    images = torch.randn(
        8,
        1,
        28,
        28,
        device=device
    )

    t = diffusion.sample_timesteps(
        images.shape[0]
    )

    x_t, noise = diffusion.noise_images(
        images,
        t
    )

    predicted_noise = model(
        x_t,
        t
    )

    loss = F.mse_loss(
        predicted_noise,
        noise
    )

    loss.backward()

    gradient_norm = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:
            gradient_norm += parameter.grad.abs().sum().item()

    assert gradient_norm > 0


def test_ddpm_loss_is_finite():
    """
    Verify that the DDPM training objective produces
    a finite numerical value.
    """

    device = "cpu"

    model = UNet().to(device)

    diffusion = Diffusion(
        device=device
    )

    images = torch.randn(
        8,
        1,
        28,
        28,
        device=device
    )

    t = diffusion.sample_timesteps(
        images.shape[0]
    )

    x_t, noise = diffusion.noise_images(
        images,
        t
    )

    predicted_noise = model(
        x_t,
        t
    )

    loss = F.mse_loss(
        predicted_noise,
        noise
    )

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)