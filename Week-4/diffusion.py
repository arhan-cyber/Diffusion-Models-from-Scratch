import torch


class Diffusion:
    """
    DDPM diffusion process implementation.

    Supports:
    - Algorithm 1 (training / forward diffusion)
    - Algorithm 2 (sampling / reverse diffusion)

    Reference:
        Denoising Diffusion Probabilistic Models
        Ho et al., 2020
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

        self.beta = (
            self.prepare_noise_schedule()
            .to(device)
        )

        self.alpha = (
            1.0 - self.beta
        )

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
        Sample timesteps uniformly.

        Corresponds to:

            t ~ Uniform({1,...,T})
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

        Equation:

            x_t =
                sqrt(alpha_hat_t) * x_0
                +
                sqrt(1 - alpha_hat_t) * epsilon

        Returns:
            x_t
            epsilon
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
            +
            sqrt_one_minus_alpha_hat * noise
        )

        return x_t, noise

    @torch.no_grad()
    def sample(
        self,
        model,
        n,
        image_size=28,
        channels=1
    ):
        """
        Generate samples using DDPM Algorithm 2.

        Args:
            model:
                Trained UNet.

            n:
                Number of samples to generate.

            image_size:
                Image height/width.

            channels:
                Number of image channels.

        Returns:
            Tensor of generated images
            in [0, 1].
        """

        model.eval()

        x = torch.randn(
            n,
            channels,
            image_size,
            image_size,
            device=self.device
        )

        for timestep in reversed(
            range(
                1,
                self.noise_steps
            )
        ):

            t = torch.full(
                (n,),
                timestep,
                device=self.device,
                dtype=torch.long
            )

            predicted_noise = model(
                x,
                t
            )

            alpha = self.alpha[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.beta[t]

            alpha = alpha[
                :,
                None,
                None,
                None
            ]

            alpha_hat = alpha_hat[
                :,
                None,
                None,
                None
            ]

            beta = beta[
                :,
                None,
                None,
                None
            ]

            if timestep > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                (
                    1.0
                    / torch.sqrt(alpha)
                )
                *
                (
                    x
                    -
                    (
                        (
                            1.0 - alpha
                        )
                        /
                        torch.sqrt(
                            1.0 - alpha_hat
                        )
                    )
                    *
                    predicted_noise
                )
                +
                torch.sqrt(beta)
                *
                noise
            )

        model.train()

        # Convert from [-1, 1]-ish range
        # back into [0, 1] for visualization.
        x = (
            x.clamp(-1, 1)
            + 1
        ) / 2

        return x