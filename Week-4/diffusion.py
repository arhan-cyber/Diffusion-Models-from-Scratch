import torch


class Diffusion:
    """
    Implements the forward diffusion process used during DDPM training.

    References:
        - DDPM Paper (Ho et al., 2020)
        - Algorithm 1
        - Equation:
            x_t = sqrt(alpha_hat_t) * x_0
                  + sqrt(1 - alpha_hat_t) * epsilon
    """

    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cpu"
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)

        self.alpha = 1.0 - self.beta

        self.alpha_hat = torch.cumprod(
            self.alpha,
            dim=0
        )

    def prepare_noise_schedule(self):
        """
        Linear beta schedule from the DDPM paper.
        """

        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.noise_steps
        )

    def sample_timesteps(
        self,
        batch_size
    ):
        """
        Sample timesteps uniformly from [1, T).

        Corresponds to:
            t ~ Uniform({1, ..., T})
        """

        return torch.randint(
            low=1,
            high=self.noise_steps,
            size=(batch_size,),
            device=self.device
        )

    def noise_images(
        self,
        x,
        t
    ):
        """
        Sample x_t from q(x_t | x_0).

        Returns:
            x_t
            noise
        """

        noise = torch.randn_like(x)

        sqrt_alpha_hat = torch.sqrt(
            self.alpha_hat[t]
        )[:, None, None, None]

        sqrt_one_minus_alpha_hat = torch.sqrt(
            1.0 - self.alpha_hat[t]
        )[:, None, None, None]

        x_t = (
            sqrt_alpha_hat * x
            + sqrt_one_minus_alpha_hat * noise
        )

        return x_t, noise