import torch


class DDIMScheduler:
    """
    DDIM Scheduler for faster sampling.
    Can be used with any pre-trained DDPM model.
    """

    def __init__(
        self,
        diffusion_obj,  # The original Week 4 Diffusion object to get beta/alpha schedules
        device="cpu"
    ):
        self.device = device
        # Ensure correct device and type
        self.alpha_hat = diffusion_obj.alpha_hat.to(device)
        self.noise_steps = diffusion_obj.noise_steps

    def get_timesteps(self, num_inference_steps):
        """
        Compute the subset of timesteps evenly spaced from [0, T-1].
        Returns a tensor of timesteps sorted in descending order.
        """
        # Linear spacing from 0 to T-1
        timesteps = torch.linspace(
            0,
            self.noise_steps - 1,
            num_inference_steps,
            dtype=torch.long,
            device=self.device
        )
        # Sort/flip to ensure descending order for reverse process
        timesteps = torch.flip(timesteps, dims=[0])
        return timesteps

    @torch.no_grad()
    def sample_step(
        self,
        model,
        x,
        t,
        prev_t,
        eta=0.0
    ):
        """
        Perform a single DDIM step from x_t to x_{prev_t}.
        
        Args:
            model: The neural network predicting noise.
            x: Current latent state at step t.
            t: Current timestep tensor (batch_size,).
            prev_t: Previous timestep tensor (batch_size,), or -1.
            eta: Stochasticity parameter (0.0 = deterministic DDIM, 1.0 = DDPM equivalent).
        """
        # Predict noise
        predicted_noise = model(x, t)

        # Get alpha values
        alpha_t = self.alpha_hat[t].view(-1, 1, 1, 1)
        
        # If prev_t is -1 (meaning we are going to t=0 / x0), we set alpha_prev = 1.0
        if isinstance(prev_t, int) and prev_t == -1:
            alpha_prev = torch.ones_like(alpha_t)
        elif torch.is_tensor(prev_t) and (prev_t < 0).all():
            alpha_prev = torch.ones_like(alpha_t)
        else:
            alpha_prev = self.alpha_hat[prev_t].view(-1, 1, 1, 1)

        # Compute sigma_t (Equation 16 of Song et al.)
        # sigma_t = eta * sqrt((1 - alpha_prev) / (1 - alpha_t)) * sqrt(1 - alpha_t / alpha_prev)
        # Note: if alpha_prev is 1.0, sigma_t is 0
        sigma_t = (
            eta
            * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t))
            * torch.sqrt(1.0 - alpha_t / alpha_prev)
        )
        # For numerical stability/safety
        sigma_t = torch.where(alpha_prev >= 1.0, torch.zeros_like(sigma_t), sigma_t)

        # Estimate x0 (Equation 12: first term)
        pred_x0 = (x - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

        # Compute direction pointing to x_t (Equation 12: second term)
        dir_xt = torch.sqrt(1.0 - alpha_prev - sigma_t ** 2) * predicted_noise

        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(x)
            # Mask noise for steps where prev_t is -1
            if isinstance(prev_t, int) and prev_t == -1:
                noise = torch.zeros_like(x)
            elif torch.is_tensor(prev_t):
                noise_mask = (prev_t >= 0).view(-1, 1, 1, 1).float()
                noise = noise * noise_mask
        else:
            noise = torch.zeros_like(x)

        # x_{prev_t}
        x_prev = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma_t * noise

        return x_prev, pred_x0

    @torch.no_grad()
    def sample(
        self,
        model,
        n,
        num_inference_steps=50,
        eta=0.0,
        image_size=28,
        channels=1,
        initial_noise=None
    ):
        """
        Generate samples using DDIM sampling.
        """
        model.eval()

        if initial_noise is not None:
            x = initial_noise.clone().to(self.device)
        else:
            x = torch.randn(
                n,
                channels,
                image_size,
                image_size,
                device=self.device
            )

        timesteps = self.get_timesteps(num_inference_steps)

        for i in range(len(timesteps)):
            t_val = timesteps[i]
            t = torch.full((n,), t_val, device=self.device, dtype=torch.long)
            
            if i + 1 < len(timesteps):
                prev_t_val = timesteps[i + 1]
                prev_t = torch.full((n,), prev_t_val, device=self.device, dtype=torch.long)
            else:
                prev_t = -1

            x, _ = self.sample_step(
                model=model,
                x=x,
                t=t,
                prev_t=prev_t,
                eta=eta
            )

        model.train()

        # Convert from [-1, 1]-ish range back into [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        return x
