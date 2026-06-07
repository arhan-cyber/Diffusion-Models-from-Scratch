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
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if timesteps < 1:
            raise ValueError(
                "timesteps must be >= 1"
            )

        self.timesteps = timesteps
        self.schedule = schedule
        self.device = torch.device(device)
        self.dtype = dtype

        # -------------------------
        # Beta schedule
        # -------------------------

        if schedule == "linear":
            betas = self._linear_beta_schedule(
                beta_start,
                beta_end,
            )
            
            # Force pure image at t=0
            betas[0] = 0.0
            self.betas = betas

        elif schedule == "cosine":
            betas = self._cosine_beta_schedule()

            # Force pure image at t=0
            betas[0] = 0.0
            self.betas = betas

        else:
            raise ValueError(
                f"Unknown schedule '{schedule}'. "
                f"Choose from ['linear', 'cosine']."
            )

        # Updated validation to allow betas[0] == 0.0
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

        # Adding a small epsilon to prevent division by zero at t=0
        self.inv_sqrt_alphas = torch.rsqrt(
            self.alphas + 1e-8
        )

    def _linear_beta_schedule(
        self,
        beta_start: float,
        beta_end: float,
    ) -> torch.Tensor:
        """
        Linear schedule from DDPM.
        """

        return torch.linspace(
            beta_start,
            beta_end,
            self.timesteps,
            dtype=self.dtype,
            device=self.device,
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
        """

        steps = self.timesteps + 1

        x = torch.linspace(
            0,
            self.timesteps,
            steps,
            dtype=self.dtype,
            device=self.device,
        )

        alpha_bar = torch.cos(
            (
                ((x / self.timesteps) + s)
                / (1 + s)
            )
            * torch.pi
            * 0.5
        ) ** 2

        alpha_bar = (
            alpha_bar / alpha_bar[0]
        )

        betas = (
            1
            - (
                alpha_bar[1:]
                / alpha_bar[:-1]
            )
        )

        return torch.clamp(
            betas,
            min=1e-8,
            max=0.999,
        )

    def _extract(
        self,
        values: torch.Tensor,
        t: torch.Tensor,
        x_shape: torch.Size,
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

        out = values[t]

        return out.view(
            t.shape[0],
            *((1,) * (len(x_shape) - 1)),
        )

    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample q(x_t | x_0) using the reparameterization trick.

        Closed-form single-step noising:
            x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

        Args:
            x0:
                Clean samples.
                Shape: [B, ...]

            t:
                Timestep indices.
                Shape: [B]

            noise:
                Optional pre-generated noise.
                If None, sampled from N(0, I).

        Returns:
            xt:
                Noisy sample at timestep t.
                Shape: [B, ...]

            noise:
                Noise used (for reference/debugging).
                Shape: [B, ...]
        """

        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self._extract(
            self.sqrt_alpha_bars,
            t,
            x0.shape,
        )

        sqrt_one_minus_alpha_bar_t = (
            self._extract(
                self.sqrt_one_minus_alpha_bars,
                t,
                x0.shape,
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
    ):
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
    ) -> torch.Tensor:
        """
        Uniformly sample timesteps.
        
        Args:
            batch_size: Number of timesteps to sample
            
        Returns:
            Tensor of shape [batch_size] with timestep indices
        """

        return torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch_size,),
            device=self.device,
        )

    def get_snr(
        self,
    ) -> torch.Tensor:
        """
        Signal-to-noise ratio across all timesteps.

        SNR(t) = α_bar(t) / (1 - α_bar(t))

        Returns:
            Tensor of shape [timesteps] with SNR values
        """

        eps = 1e-8

        return self.alpha_bars / (
            1.0
            - self.alpha_bars
            + eps
        )

    def to(
        self,
        device: str | torch.device,
    ):
        """
        Move scheduler tensors to a new device.
        
        Args:
            device: Target device (e.g., 'cpu', 'cuda', 'cuda:0')
            
        Returns:
            Self (for method chaining)
        """

        device = torch.device(device)

        self.device = torch.device(device.type)

        for name, value in vars(self).items():
            if isinstance(
                value,
                torch.Tensor,
            ):
                setattr(
                    self,
                    name,
                    value.to(device),
                )

        return self