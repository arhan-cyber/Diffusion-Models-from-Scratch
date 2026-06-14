import torch
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)


from diffusion import Diffusion
from model import UNet


def test_sampling_shape():

    model = UNet()

    diffusion = Diffusion()

    samples = diffusion.sample(
        model=model,
        n=4
    )

    assert samples.shape == (
        4,
        1,
        28,
        28
    )


def test_sampling_contains_no_nan():

    model = UNet()

    diffusion = Diffusion()

    samples = diffusion.sample(
        model=model,
        n=2
    )

    assert not torch.isnan(
        samples
    ).any()


def test_sampling_range():

    model = UNet()

    diffusion = Diffusion()

    samples = diffusion.sample(
        model=model,
        n=2
    )

    assert samples.min() >= 0.0
    assert samples.max() <= 1.0