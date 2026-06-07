"""
Unit tests for NoiseScheduler forward diffusion process.

Tests:
    - Linear and cosine beta schedules
    - Closed-form q(x_t | x_0) sampling
    - Mathematical correctness of precomputed quantities
    - Convergence toward N(0, I) at large timesteps
"""

import pytest
import torch

from scheduler import NoiseScheduler


class TestNoiseSchedulerInitialization:
    """Test scheduler initialization and validation."""

    def test_linear_schedule_initialization(self):
        scheduler = NoiseScheduler(
            timesteps=1000,
            schedule="linear",
        )

        assert scheduler.timesteps == 1000
        assert scheduler.schedule == "linear"
        assert scheduler.betas.shape == (1000,)

    def test_cosine_schedule_initialization(self):
        scheduler = NoiseScheduler(
            timesteps=1000,
            schedule="cosine",
        )

        assert scheduler.timesteps == 1000
        assert scheduler.schedule == "cosine"
        assert scheduler.betas.shape == (1000,)

    def test_invalid_schedule_raises_error(self):
        with pytest.raises(
            ValueError,
            match="Unknown schedule",
        ):
            NoiseScheduler(
                schedule="invalid",
            )

    def test_invalid_timesteps_raises_error(self):
        with pytest.raises(
            ValueError,
            match="timesteps must be >= 1",
        ):
            NoiseScheduler(
                timesteps=0,
            )

    def test_beta_bounds_validation(self):
        """
        beta[0] is intentionally 0.0.

        All betas should lie in [0, 1).
        """

        for schedule in [
            "linear",
            "cosine",
        ]:
            scheduler = NoiseScheduler(
                schedule=schedule,
            )

            assert torch.all(
                scheduler.betas >= 0
            )

            assert torch.all(
                scheduler.betas < 1
            )

    def test_precomputed_quantities_exist(self):
        scheduler = NoiseScheduler()

        required_attrs = [
            "betas",
            "alphas",
            "alpha_bars",
            "sqrt_alphas",
            "sqrt_alpha_bars",
            "sqrt_one_minus_alpha_bars",
        ]

        for attr in required_attrs:
            assert hasattr(
                scheduler,
                attr,
            )

            assert isinstance(
                getattr(
                    scheduler,
                    attr,
                ),
                torch.Tensor,
            )

    def test_t0_is_clean_image(self):
        scheduler = NoiseScheduler()

        assert scheduler.betas[0] == 0.0

        assert torch.allclose(
            scheduler.alpha_bars[0],
            torch.tensor(
                1.0,
                dtype=scheduler.alpha_bars.dtype,
            ),
            atol=1e-6,
        )


class TestAlphaBarProperties:
    """Test alpha_bar mathematical properties."""

    @pytest.mark.parametrize(
        "schedule",
        [
            "linear",
            "cosine",
        ],
    )
    def test_alpha_bar_monotonic_decreasing(
        self,
        schedule,
    ):
        scheduler = NoiseScheduler(
            schedule=schedule,
        )

        assert scheduler.alpha_bars[
            0
        ] == pytest.approx(
            1.0,
            abs=1e-6,
        )

        assert (
            scheduler.alpha_bars[-1]
            < 0.01
        )

        diffs = torch.diff(
            scheduler.alpha_bars
        )

        assert torch.all(
            diffs <= 0
        )

    @pytest.mark.parametrize(
        "schedule",
        [
            "linear",
            "cosine",
        ],
    )
    def test_alpha_cumprod_property(
        self,
        schedule,
    ):
        scheduler = NoiseScheduler(
            schedule=schedule,
        )

        expected = torch.cumprod(
            scheduler.alphas,
            dim=0,
        )

        assert torch.allclose(
            scheduler.alpha_bars,
            expected,
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        "schedule",
        [
            "linear",
            "cosine",
        ],
    )
    def test_precomputed_sqrt_consistency(
        self,
        schedule,
    ):
        scheduler = NoiseScheduler(
            schedule=schedule,
        )

        sum_of_squares = (
            scheduler.sqrt_alpha_bars**2
            + scheduler.sqrt_one_minus_alpha_bars**2
        )

        assert torch.allclose(
            sum_of_squares,
            torch.ones_like(
                sum_of_squares
            ),
            atol=1e-5,
        )


class TestAddNoiseReparameterization:
    """Test q(x_t | x_0)."""

    def test_add_noise_shape_preservation(
        self,
    ):
        scheduler = NoiseScheduler()

        shapes = [
            (2, 3, 32, 32),
            (4, 64),
            (8, 3, 16, 16),
        ]

        for shape in shapes:
            x0 = torch.randn(
                shape
            )

            t = torch.randint(
                0,
                scheduler.timesteps,
                (shape[0],),
            )

            xt, noise = (
                scheduler.add_noise(
                    x0,
                    t,
                )
            )

            assert (
                xt.shape
                == x0.shape
            )

            assert (
                noise.shape
                == x0.shape
            )

    def test_add_noise_deterministic_with_seed(
        self,
    ):
        x0 = torch.randn(
            4,
            3,
            32,
            32,
        )

        t = torch.tensor(
            [
                100,
                250,
                500,
                750,
            ]
        )

        torch.manual_seed(
            42
        )

        scheduler1 = (
            NoiseScheduler()
        )

        xt1, noise1 = (
            scheduler1.add_noise(
                x0,
                t,
            )
        )

        torch.manual_seed(
            42
        )

        scheduler2 = (
            NoiseScheduler()
        )

        xt2, noise2 = (
            scheduler2.add_noise(
                x0,
                t,
            )
        )

        assert torch.allclose(
            xt1,
            xt2,
            atol=1e-6,
        )

        assert torch.allclose(
            noise1,
            noise2,
            atol=1e-6,
        )

    def test_add_noise_with_preset_noise(
        self,
    ):
        scheduler = (
            NoiseScheduler()
        )

        x0 = torch.randn(
            2,
            3,
            32,
            32,
        )

        t = torch.tensor(
            [100, 250]
        )

        noise = torch.randn_like(
            x0
        )

        xt, returned_noise = (
            scheduler.add_noise(
                x0,
                t,
                noise=noise,
            )
        )

        assert torch.allclose(
            returned_noise,
            noise,
        )

        sqrt_ab_t = (
            scheduler._extract(
                scheduler.sqrt_alpha_bars,
                t,
                x0,
            )
        )

        sqrt_one_minus_ab_t = (
            scheduler._extract(
                scheduler.sqrt_one_minus_alpha_bars,
                t,
                x0,
            )
        )

        expected_xt = (
            sqrt_ab_t * x0
            + sqrt_one_minus_ab_t
            * noise
        )

        assert torch.allclose(
            xt,
            expected_xt,
            atol=1e-5,
        )

    def test_q_sample_alias(self):
        scheduler = (
            NoiseScheduler()
        )

        x0 = torch.randn(
            2,
            3,
            32,
            32,
        )

        t = torch.tensor(
            [100, 250]
        )

        torch.manual_seed(
            42
        )

        xt1, noise1 = (
            scheduler.add_noise(
                x0,
                t,
            )
        )

        torch.manual_seed(
            42
        )

        xt2, noise2 = (
            scheduler.q_sample(
                x0,
                t,
            )
        )

        assert torch.allclose(
            xt1,
            xt2,
            atol=1e-6,
        )

        assert torch.allclose(
            noise1,
            noise2,
            atol=1e-6,
        )


class TestGaussianLimitAtT:
    """Test behavior at boundary timesteps."""

    @pytest.mark.parametrize(
        "schedule",
        [
            "linear",
            "cosine",
        ],
    )
    def test_xt_approaches_gaussian_at_t_equals_T(
        self,
        schedule,
    ):
        scheduler = (
            NoiseScheduler(
                timesteps=1000,
                schedule=schedule,
            )
        )

        batch_size = 1000

        x0 = torch.randn(
            batch_size,
            3,
            32,
            32,
        )

        t = torch.full(
            (batch_size,),
            999,
            dtype=torch.long,
        )

        xt, _ = (
            scheduler.add_noise(
                x0,
                t,
            )
        )

        mean = xt.mean()
        std = xt.std()

        assert torch.allclose(
            mean,
            torch.tensor(
                0.0
            ),
            atol=0.05,
        )

        assert torch.allclose(
            std,
            torch.tensor(
                1.0
            ),
            atol=0.05,
        )

        alpha_bar_T = (
            scheduler.alpha_bars[
                -1
            ]
        )

        assert (
            alpha_bar_T
            < 1e-3
        )

    def test_xt_at_t0_preserves_x0(
        self,
    ):
        scheduler = (
            NoiseScheduler()
        )

        x0 = torch.randn(
            10,
            3,
            32,
            32,
        )

        t = torch.zeros(
            10,
            dtype=torch.long,
        )

        aggressive_noise = (
            torch.randn_like(
                x0
            )
            * 100.0
        )

        xt, _ = (
            scheduler.add_noise(
                x0,
                t,
                noise=aggressive_noise,
            )
        )

        assert torch.allclose(
            xt,
            x0,
            atol=1e-6,
        )


class TestSNRComputation:
    """Test SNR calculations."""

    def test_snr_shape(self):
        scheduler = (
            NoiseScheduler(
                timesteps=1000,
            )
        )

        snr = (
            scheduler.get_snr()
        )

        assert snr.shape == (
            1000,
        )

    @pytest.mark.parametrize(
        "schedule",
        [
            "linear",
            "cosine",
        ],
    )
    def test_snr_monotonic_decreasing(
        self,
        schedule,
    ):
        scheduler = (
            NoiseScheduler(
                schedule=schedule,
            )
        )

        snr = (
            scheduler.get_snr()
        )

        diffs = torch.diff(
            snr
        )

        assert torch.all(
            diffs <= 0
        )

    def test_snr_formula(self):
        scheduler = (
            NoiseScheduler()
        )

        snr = (
            scheduler.get_snr()
        )

        expected = (
            scheduler.alpha_bars
            / (
                1.0
                - scheduler.alpha_bars
                + 1e-8
            )
        )

        assert torch.allclose(
            snr[1:-1],
            expected[1:-1],
            rtol=1e-4,
        )

    def test_add_noise_respects_device(
        self,
    ):
        scheduler = (
            NoiseScheduler()
        )

        x0 = torch.randn(
            2,
            3,
            32,
            32,
        )

        t = torch.tensor(
            [100, 250]
        )

        xt, noise = (
            scheduler.add_noise(
                x0,
                t,
            )
        )

        assert (
            xt.device
            == x0.device
        )

        assert (
            noise.device
            == x0.device
        )


class TestScheduleComparison:
    """Compare linear and cosine schedules."""

    def test_linear_vs_cosine_beta_values(
        self,
    ):
        linear = (
            NoiseScheduler(
                schedule="linear",
            )
        )

        cosine = (
            NoiseScheduler(
                schedule="cosine",
            )
        )

        assert not torch.allclose(
            linear.betas,
            cosine.betas,
        )

    def test_cosine_snr_decays_smoother(
        self,
    ):
        linear = (
            NoiseScheduler(
                schedule="linear",
            )
        )

        cosine = (
            NoiseScheduler(
                schedule="cosine",
            )
        )

        snr_linear = (
            linear.get_snr()
        )

        snr_cosine = (
            cosine.get_snr()
        )

        mid = 500

        assert (
            snr_cosine[mid]
            > snr_linear[mid]
        )


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
        ]
    )