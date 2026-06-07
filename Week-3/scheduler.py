"""
DDPM-style noise scheduler for the forward diffusion process.

References:
    - Ho et al. (2020): Denoising Diffusion Probabilistic Models
    - Nichol & Dhariwal (2021): Improved Denoising Diffusion Probabilistic Models
"""

import torch


class NoiseScheduler:
    """
    DDPM-style noise scheduler.

    Supports:
        - Linear beta schedule
        - Cosine beta schedule

    References:
        - Ho et al. (2020)
        - Nichol & Dhariwal (2021)
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        dtype: torch.dtype = torch.float32,
    ):
        if timesteps < 1:
            raise ValueError(
                "timesteps must be >= 1"
            )

        self.timesteps = timesteps
        self.schedule = schedule
        self.dtype = dtype

        # -------------------------
        # Beta schedule
        # -------------------------

        if schedule == "linear":
            self.betas = self._linear_beta_schedule(
                beta_start,
                beta_end,
            )

        elif schedule == "cosine":
            self.betas = self._cosine_beta_schedule()

        else:
            raise ValueError(
                f"Unknown schedule '{schedule}'. "
                f"Choose from ['linear', 'cosine']."
            )

        if not torch.all(
            (self.betas >= 0)
            & (self.betas < 1)
        ):
            raise ValueError(
                "All beta values must lie in [0, 1)."
            )

        # -------------------------
        # Core diffusion quantities
        # -------------------------

        self.alphas = 1.0 - self.betas

        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0,
        )

        # -------------------------
        # Precomputed quantities
        # -------------------------

        self.sqrt_alphas = torch.sqrt(
            self.alphas
        )

        self.sqrt_alpha_bars = torch.sqrt(
            self.alpha_bars
        )

        self.sqrt_one_minus_alpha_bars = torch.sqrt(
            1.0 - self.alpha_bars
        )

    def _linear_beta_schedule(
        self,
        beta_start: float,
        beta_end: float,
    ) -> torch.Tensor:
        """
        Linear schedule from DDPM.

        beta[0] = 0 ensures:
            alpha_bar[0] = 1
        so t=0 corresponds to a clean image.
        """

        if self.timesteps == 1:
            return torch.zeros(
                1,
                dtype=self.dtype,
            )

        betas = torch.linspace(
            beta_start,
            beta_end,
            self.timesteps - 1,
            dtype=self.dtype,
        )

        return torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=self.dtype,
                ),
                betas,
            ]
        )

    def _cosine_beta_schedule(
        self,
        s: float = 0.008,
    ) -> torch.Tensor:
        """
        Cosine schedule from:

        Improved Denoising Diffusion
        Probabilistic Models
        (Nichol & Dhariwal, 2021)

        beta[0] = 0 ensures:
            alpha_bar[0] = 1
        so t=0 corresponds to a clean image.
        """

        if self.timesteps == 1:
            return torch.zeros(
                1,
                dtype=self.dtype,
            )

        steps = self.timesteps

        x = torch.linspace(
            0,
            steps - 1,
            steps,
            dtype=self.dtype,
        )

        alpha_bar = torch.cos(
            (
                ((x / (steps - 1)) + s)
                / (1 + s)
            )
            * torch.pi
            * 0.5
        ) ** 2

        alpha_bar = (
            alpha_bar
            / alpha_bar[0]
        )

        betas = (
            1
            - (
                alpha_bar[1:]
                / alpha_bar[:-1]
            )
        )

        betas = torch.clamp(
            betas,
            min=1e-8,
            max=0.999,
        )

        return torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=self.dtype,
                ),
                betas,
            ]
        )

    def _extract(
        self,
        values: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract values at timestep t and
        reshape for broadcasting.

        Works for arbitrary tensor ranks:
            [B, D]
            [B, C, T]
            [B, C, H, W]
            [B, C, T, H, W]
            ...
        """

        values = values.to(
            device=t.device
        )

        return values[t].reshape(
            t.shape[0],
            *((1,) * (x.ndim - 1)),
        )

    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample q(x_t | x_0) using the
        reparameterization trick.

        x_t =
            sqrt(alpha_bar_t) * x_0
            +
            sqrt(1 - alpha_bar_t) * eps

        Args:
            x0:
                Clean samples.
                Shape: [B, ...]

            t:
                Timestep indices.
                Shape: [B]

            noise:
                Optional pre-generated noise.

        Returns:
            xt:
                Noisy sample at timestep t.

            noise:
                Noise used.
        """

        if noise is None:
            noise = torch.randn_like(
                x0
            )

        sqrt_alpha_bar_t = (
            self._extract(
                self.sqrt_alpha_bars,
                t,
                x0,
            )
        )

        sqrt_one_minus_alpha_bar_t = (
            self._extract(
                self.sqrt_one_minus_alpha_bars,
                t,
                x0,
            )
        )

        xt = (
            sqrt_alpha_bar_t * x0
            + sqrt_one_minus_alpha_bar_t
            * noise
        )

        return xt, noise

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Alias used in many diffusion repos.

        Equivalent to add_noise().
        """

        return self.add_noise(
            x0,
            t,
            noise,
        )

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Uniformly sample timesteps.

        Args:
            batch_size:
                Number of timesteps.

            device:
                Optional output device.

        Returns:
            Tensor of shape [batch_size].
        """

        return torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch_size,),
            device=device,
        )

    def get_snr(
        self,
    ) -> torch.Tensor:
        """
        Signal-to-noise ratio.

        SNR(t) =
            alpha_bar(t)
            /
            (1 - alpha_bar(t))
        """

        eps = 1e-8

        return self.alpha_bars / (
            1.0
            - self.alpha_bars
            + eps
        )