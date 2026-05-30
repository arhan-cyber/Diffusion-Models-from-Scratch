from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:

    name: str

    # Model
    in_channels: int = 1
    out_channels: int = 1

    base_channels: int = 32
    depth: int = 3

    use_skip_connections: bool = True
    use_batchnorm: bool = True

    final_activation: Optional[str] = None

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50

    # Noise
    noise_min: float = 0.05
    noise_max: float = 0.50


EXPERIMENTS = [

    ExperimentConfig(
        name="baseline"
    ),

    ExperimentConfig(
        name="depth4",
        depth=4
    ),

    ExperimentConfig(
        name="depth3",
        depth=3
    ),

    ExperimentConfig(
        name="width64",
        base_channels=64
    ),

    ExperimentConfig(
        name="width128",
        base_channels=128
    ),

    ExperimentConfig(
        name="no_skip",
        use_skip_connections=False
    ),

    ExperimentConfig(
        name="no_batchnorm",
        use_batchnorm=False
    ),

    ExperimentConfig(
        name="low_noise",
        noise_min=0.05,
        noise_max=0.15
    ),

    ExperimentConfig(
        name="high_noise",
        noise_min=0.30,
        noise_max=0.70
    ),
]