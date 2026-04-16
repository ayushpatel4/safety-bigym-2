"""
Unit tests for OrnsteinUhlenbeckNoise (Phase 1 BodySLAM++ noise model).

BodySLAM++ errors are temporally correlated across frames — the pose regressor
latches onto the same slightly-wrong skeleton configuration for several frames
in a row. An i.i.d. Gaussian noise model underestimates how hard the policy
has to work to disambiguate. OrnsteinUhlenbeckNoise reproduces this
autocorrelation with a tunable alpha and a steady-state std that matches the
reported BodySLAM++ ATE (~3cm).
"""

import numpy as np
import pytest

from safety_bigym.filters.noise_models import OrnsteinUhlenbeckNoise


def _sample_many(noise, n_steps):
    return np.stack([noise.step() for _ in range(n_steps)])


class TestOUNoiseShape:
    def test_sample_shape_matches_config(self):
        noise = OrnsteinUhlenbeckNoise(shape=(18, 3), alpha=0.9, sigma=0.05, seed=0)
        assert noise.step().shape == (18, 3)

    def test_sample_shape_scalar(self):
        noise = OrnsteinUhlenbeckNoise(shape=(3,), alpha=0.9, sigma=0.05, seed=0)
        assert noise.step().shape == (3,)


class TestOUNoiseTemporalCorrelation:
    def test_lag1_autocorrelation_with_alpha_09(self):
        noise = OrnsteinUhlenbeckNoise(shape=(1,), alpha=0.9, sigma=0.05, seed=0)
        # Warm up past transient.
        for _ in range(500):
            noise.step()
        samples = _sample_many(noise, 5000).ravel()
        # Pearson correlation between x[:-1] and x[1:].
        a = samples[:-1]
        b = samples[1:]
        corr = float(np.corrcoef(a, b)[0, 1])
        assert corr > 0.8, f"expected lag-1 autocorr > 0.8, got {corr:.3f}"

    def test_iid_fallback_with_alpha_0(self):
        noise = OrnsteinUhlenbeckNoise(shape=(1,), alpha=0.0, sigma=0.1, seed=0)
        samples = _sample_many(noise, 5000).ravel()
        a = samples[:-1]
        b = samples[1:]
        corr = float(np.corrcoef(a, b)[0, 1])
        assert abs(corr) < 0.1, f"alpha=0 must be near-uncorrelated, got {corr:.3f}"


class TestOUNoiseSteadyState:
    def test_std_matches_sigma_after_warmup(self):
        sigma = 0.05
        noise = OrnsteinUhlenbeckNoise(shape=(1,), alpha=0.9, sigma=sigma, seed=0)
        for _ in range(500):
            noise.step()
        samples = _sample_many(noise, 20_000).ravel()
        measured = float(samples.std())
        assert measured == pytest.approx(sigma, rel=0.1), (
            f"steady-state std {measured:.4f} should equal sigma {sigma:.4f} within 10%"
        )

    def test_mean_is_zero(self):
        noise = OrnsteinUhlenbeckNoise(shape=(1,), alpha=0.9, sigma=0.05, seed=0)
        samples = _sample_many(noise, 20_000).ravel()
        assert abs(samples.mean()) < 0.01


class TestOUNoiseReproducibility:
    def test_same_seed_produces_identical_sequence(self):
        n1 = OrnsteinUhlenbeckNoise(shape=(18, 3), alpha=0.9, sigma=0.05, seed=42)
        n2 = OrnsteinUhlenbeckNoise(shape=(18, 3), alpha=0.9, sigma=0.05, seed=42)
        a = _sample_many(n1, 100)
        b = _sample_many(n2, 100)
        np.testing.assert_allclose(a, b)

    def test_different_seeds_diverge(self):
        n1 = OrnsteinUhlenbeckNoise(shape=(18, 3), alpha=0.9, sigma=0.05, seed=1)
        n2 = OrnsteinUhlenbeckNoise(shape=(18, 3), alpha=0.9, sigma=0.05, seed=2)
        a = _sample_many(n1, 100)
        b = _sample_many(n2, 100)
        assert not np.allclose(a, b)

    def test_reset_with_seed_restores_sequence(self):
        noise = OrnsteinUhlenbeckNoise(shape=(3,), alpha=0.9, sigma=0.05, seed=7)
        first = _sample_many(noise, 50)
        noise.reset(seed=7)
        second = _sample_many(noise, 50)
        np.testing.assert_allclose(first, second)


class TestOUNoiseResetClearsState:
    def test_reset_zeros_internal_state(self):
        noise = OrnsteinUhlenbeckNoise(shape=(3,), alpha=0.99, sigma=1.0, seed=0)
        # Drive internal state far from zero.
        for _ in range(2000):
            noise.step()
        noise.reset(seed=None)
        np.testing.assert_array_equal(noise.state, np.zeros(3))
