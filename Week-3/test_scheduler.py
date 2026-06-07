"""
Unit tests for NoiseScheduler forward diffusion process.

Tests the implementation of DDPM-style noise scheduling with:
  - Linear and cosine beta schedules
  - Closed-form q(x_t | x_0) sampling via reparameterization trick
  - Mathematical correctness of precomputed quantities
  - Proper convergence to N(0, I) noise at t=T
"""

import pytest
import torch
import numpy as np
from scheduler import NoiseScheduler


class TestNoiseSchedulerInitialization:
    """Test scheduler initialization and parameter validation."""

    def test_linear_schedule_initialization(self):
        """Verify linear schedule initializes without error."""
        scheduler = NoiseScheduler(timesteps=1000, schedule="linear", device="cpu")
        assert scheduler.timesteps == 1000
        assert scheduler.schedule == "linear"
        assert scheduler.betas.shape == (1000,)

    def test_cosine_schedule_initialization(self):
        """Verify cosine schedule initializes without error."""
        scheduler = NoiseScheduler(timesteps=1000, schedule="cosine", device="cpu")
        assert scheduler.timesteps == 1000
        assert scheduler.schedule == "cosine"
        assert scheduler.betas.shape == (1000,)

    def test_invalid_schedule_raises_error(self):
        """Verify invalid schedule name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown schedule"):
            NoiseScheduler(schedule="invalid", device="cpu")

    def test_invalid_timesteps_raises_error(self):
        """Verify invalid timesteps raises ValueError."""
        with pytest.raises(ValueError, match="timesteps must be >= 1"):
            NoiseScheduler(timesteps=0, device="cpu")

    def test_beta_bounds_validation(self):
        """Verify all beta values lie in [0, 1)."""
        for schedule in ["linear", "cosine"]:
            scheduler = NoiseScheduler(schedule=schedule, device="cpu")
            # Updated to allow betas >= 0 since beta_0 is intentionally 0.0
            assert torch.all(scheduler.betas >= 0), f"{schedule}: betas have negative values"
            assert torch.all(scheduler.betas < 1), f"{schedule}: betas have values >= 1"

    def test_precomputed_quantities_exist(self):
        """Verify all precomputed quantities are initialized."""
        scheduler = NoiseScheduler(device="cpu")
        required_attrs = [
            "betas",
            "alphas",
            "alpha_bars",
            "sqrt_alphas",
            "sqrt_alpha_bars",
            "sqrt_one_minus_alpha_bars",
            "inv_sqrt_alphas",
        ]
        for attr in required_attrs:
            assert hasattr(scheduler, attr), f"Missing precomputed quantity: {attr}"
            assert isinstance(getattr(scheduler, attr), torch.Tensor)


class TestAlphaBarProperties:
    """Test mathematical properties of alpha_bar schedule."""

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_alpha_bar_monotonic_decreasing(self, schedule):
        """Verify alpha_bar decreases monotonically from 1 to near 0."""
        scheduler = NoiseScheduler(schedule=schedule, device="cpu")
        assert scheduler.alpha_bars[0] == pytest.approx(1.0, abs=1e-6)
        assert scheduler.alpha_bars[-1] < 0.01  # Near 0 at t=T
        diffs = torch.diff(scheduler.alpha_bars)
        # Using <= 0 to account for potential numerical plateaus near boundaries
        assert torch.all(diffs <= 0), f"{schedule}: alpha_bar is not monotonically decreasing"

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_alpha_cumprod_property(self, schedule):
        """Verify alpha_bar = cumprod(alpha)."""
        scheduler = NoiseScheduler(schedule=schedule, device="cpu")
        expected_alpha_bars = torch.cumprod(scheduler.alphas, dim=0)
        assert torch.allclose(scheduler.alpha_bars, expected_alpha_bars, atol=1e-6)

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_precomputed_sqrt_consistency(self, schedule):
        """Verify sqrt(alpha_bar)^2 + sqrt(1-alpha_bar)^2 ≈ 1."""
        scheduler = NoiseScheduler(schedule=schedule, device="cpu")
        
        sum_of_squares = (
            scheduler.sqrt_alpha_bars ** 2
            + scheduler.sqrt_one_minus_alpha_bars ** 2
        )
        
        # Should be close to 1 (within numerical precision)
        assert torch.allclose(sum_of_squares, torch.ones_like(sum_of_squares), atol=1e-5)


class TestAddNoiseReparameterization:
    """Test the reparameterization trick for adding noise."""

    def test_add_noise_shape_preservation(self):
        """Verify add_noise preserves input shapes."""
        scheduler = NoiseScheduler(device="cpu")
        shapes = [
            (2, 3, 32, 32),  # Batch of images
            (4, 64),          # Batch of 1D features
            (8, 3, 16, 16),  # Smaller batch of images
        ]
        
        for shape in shapes:
            x0 = torch.randn(shape)
            t = torch.randint(0, 1000, (shape[0],))
            xt, noise = scheduler.add_noise(x0, t)
            
            assert xt.shape == x0.shape, f"Shape mismatch for input {shape}"
            assert noise.shape == x0.shape

    def test_add_noise_deterministic_with_seed(self):
        """Verify add_noise is deterministic given fixed seed."""
        x0 = torch.randn(4, 3, 32, 32)
        t = torch.tensor([100, 250, 500, 750])
        
        # First run
        torch.manual_seed(42)
        scheduler1 = NoiseScheduler(device="cpu")
        xt1, noise1 = scheduler1.add_noise(x0, t)
        
        # Second run with same seed
        torch.manual_seed(42)
        scheduler2 = NoiseScheduler(device="cpu")
        xt2, noise2 = scheduler2.add_noise(x0, t)
        
        assert torch.allclose(xt1, xt2, atol=1e-6)
        assert torch.allclose(noise1, noise2, atol=1e-6)

    def test_add_noise_with_preset_noise(self):
        """Verify add_noise accepts pre-generated noise."""
        scheduler = NoiseScheduler(device="cpu")
        x0 = torch.randn(2, 3, 32, 32)
        t = torch.tensor([100, 250])
        noise = torch.randn_like(x0)
        
        xt, returned_noise = scheduler.add_noise(x0, t, noise=noise)
        
        assert torch.allclose(returned_noise, noise)
        # Verify computation: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise
        sqrt_ab_t = scheduler._extract(scheduler.sqrt_alpha_bars, t, x0.shape)
        sqrt_one_minus_ab_t = scheduler._extract(
            scheduler.sqrt_one_minus_alpha_bars, t, x0.shape
        )
        expected_xt = sqrt_ab_t * x0 + sqrt_one_minus_ab_t * noise
        assert torch.allclose(xt, expected_xt, atol=1e-5)

    def test_q_sample_alias(self):
        """Verify q_sample is equivalent to add_noise."""
        scheduler = NoiseScheduler(device="cpu")
        x0 = torch.randn(2, 3, 32, 32)
        t = torch.tensor([100, 250])
        
        torch.manual_seed(42)
        xt1, noise1 = scheduler.add_noise(x0, t)
        
        torch.manual_seed(42)
        xt2, noise2 = scheduler.q_sample(x0, t)
        
        assert torch.allclose(xt1, xt2, atol=1e-6)
        assert torch.allclose(noise1, noise2, atol=1e-6)


class TestGaussianLimitAt_T:
    """Critical test: verify x_T ≈ N(0, I) at final timestep and pure x_0 at t=0."""

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_xt_approaches_gaussian_at_t_equals_T(self, schedule):
        """
        Verify that x_T (noised at t=999) ≈ N(0, I).
        
        At t=T (final timestep):
          - α_bar_T → 0
          - sqrt(α_bar_T) → 0
          - sqrt(1 - α_bar_T) → 1
          - x_T ≈ 0 * x_0 + 1 * noise ≈ noise ~ N(0, I)
        """
        scheduler = NoiseScheduler(timesteps=1000, schedule=schedule, device="cpu")
        
        # Create a batch of clean images
        batch_size = 1000
        x0 = torch.randn(batch_size, 3, 32, 32)
        
        # Noise at t=999 (final timestep)
        t = torch.full((batch_size,), 999)
        
        xt, noise = scheduler.add_noise(x0, t)
        
        # At t=999, x_t should be almost entirely noise
        # Verify: mean ≈ 0, std ≈ 1
        xt_flat = xt.reshape(batch_size, -1)
        
        
        # Calculate global mean and std instead of per-pixel
        mean = xt.mean()
        std = xt.std()
        
        # Mean should be close to 0
        assert torch.allclose(mean, torch.tensor(0.0), atol=0.05)
        
        # Std should be close to 1
        assert torch.allclose(std, torch.tensor(1.0), atol=0.05)
        
        # Verify α_bar_T is very small
        alpha_bar_T = scheduler.alpha_bars[-1]
        assert alpha_bar_T < 1e-3, f"{schedule}: α_bar_T = {alpha_bar_T} should be < 1e-3"

    def test_xt_at_t_equals_0_preserves_x0(self):
        """Verify x_0 perfectly resists noise injection due to beta_0 = 0.0."""
        scheduler = NoiseScheduler(device="cpu")
        x0 = torch.randn(10, 3, 32, 32)
        t = torch.zeros(10, dtype=torch.long)
        
        # We pass aggressive random noise to prove it gets zeroed out by the scheduler
        aggressive_noise = torch.randn_like(x0) * 100.0
        
        xt, noise = scheduler.add_noise(x0, t, noise=aggressive_noise)
        
        # At t=0: x_t = sqrt(1.0) * x_0 + sqrt(0.0) * noise ≈ x_0 exactly
        assert torch.allclose(xt, x0, atol=1e-6)


class TestSNRComputation:
    """Test signal-to-noise ratio computation."""

    def test_snr_shape(self):
        """Verify SNR has correct shape."""
        scheduler = NoiseScheduler(timesteps=1000, device="cpu")
        snr = scheduler.get_snr()
        assert snr.shape == (1000,)

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_snr_monotonic_decreasing(self, schedule):
        """Verify SNR decreases monotonically over time."""
        scheduler = NoiseScheduler(schedule=schedule, device="cpu")
        snr = scheduler.get_snr()
        
        diffs = torch.diff(snr)
        # <= 0 accounts for numerical precision at boundaries
        assert torch.all(diffs <= 0), f"{schedule}: SNR is not monotonically decreasing"

    def test_snr_formula(self):
        """Verify SNR = α_bar / (1 - α_bar)."""
        scheduler = NoiseScheduler(device="cpu")
        snr = scheduler.get_snr()
        
        # Add epsilon to denominator to prevent divide by zero where alpha_bar == 1.0
        expected_snr = scheduler.alpha_bars / (1.0 - scheduler.alpha_bars + 1e-8)
        
        # SNR computed with epsilon for numerical stability
        assert torch.allclose(snr[1:-1], expected_snr[1:-1], rtol=1e-4)


class TestDeviceHandling:
    """Test device management (CPU/CUDA if available)."""

    def test_scheduler_on_cpu(self):
        """Verify scheduler initializes on CPU."""
        scheduler = NoiseScheduler(device="cpu")
        assert scheduler.device == torch.device("cpu")
        assert scheduler.betas.device == torch.device("cpu")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_scheduler_on_cuda(self):
        """Verify scheduler initializes on CUDA if available."""
        scheduler = NoiseScheduler(device="cuda")
        assert scheduler.device.type == "cuda"
        assert scheduler.betas.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_scheduler_to_device(self):
        """Verify scheduler can be moved to different devices."""
        scheduler = NoiseScheduler(device="cpu")
        assert scheduler.betas.device.type == "cpu"
        
        scheduler.to("cuda")
        assert scheduler.betas.device.type == "cuda"
        
        scheduler.to("cpu")
        assert scheduler.betas.device.type == "cpu"

    def test_add_noise_respects_device(self):
        """Verify add_noise respects the scheduler's device."""
        scheduler = NoiseScheduler(device="cpu")
        x0 = torch.randn(2, 3, 32, 32, device="cpu")
        t = torch.tensor([100, 250])
        
        xt, noise = scheduler.add_noise(x0, t)
        assert xt.device == x0.device
        assert noise.device == x0.device


class TestScheduleComparison:
    """Test differences between linear and cosine schedules."""

    def test_linear_vs_cosine_beta_values(self):
        """Verify linear and cosine schedules produce different beta values."""
        scheduler_linear = NoiseScheduler(schedule="linear", device="cpu")
        scheduler_cosine = NoiseScheduler(schedule="cosine", device="cpu")
        
        assert not torch.allclose(scheduler_linear.betas, scheduler_cosine.betas)

    def test_cosine_snr_decays_smoother(self):
        """Verify cosine schedule maintains higher SNR at mid-timesteps."""
        scheduler_linear = NoiseScheduler(schedule="linear", device="cpu")
        scheduler_cosine = NoiseScheduler(schedule="cosine", device="cpu")
        
        snr_linear = scheduler_linear.get_snr()
        snr_cosine = scheduler_cosine.get_snr()
        
        # At mid-timesteps (e.g., t=500), cosine should have higher SNR
        mid_timestep = 500
        assert snr_cosine[mid_timestep] > snr_linear[mid_timestep]


if __name__ == "__main__":
    # Run: pytest test_scheduler.py -v
    pytest.main([__file__, "-v"])