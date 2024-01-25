import os

import numpy as np
import pytest
from scipy.stats import normaltest, pearsonr

import geomstats.backend as gs
from geomstats.distributions.brownian_motion import BrownianMotion
from geomstats.geometry.euclidean import Euclidean
from geomstats.test.test_case import TestCase


@pytest.mark.skipif(
    os.environ.get("GEOMSTATS_BACKEND") != "pytorch",
    reason="BrownianMotion requires pytorch backend.",
)
class TestBrownianMotion(TestCase):
    """Verify the axioms of Brownian Motion"""

    def setup_method(self):
        self.end_time = 1
        self.num_steps = 50
        self.dimension = 2
        self.manifold = Euclidean(self.dimension)
        self.brownian_motion = BrownianMotion(self.manifold)
        self.num_samples = 1000
        self.samples = self._test_sample()

    def _test_sample(self):
        r"""
        Generate samples of Brownian motion.
        Use a fixed seed to ensure reproducibility.
        """
        np.random.seed(0)
        samples = np.zeros((self.num_samples, self.num_steps, self.dimension))
        for i in range(self.num_samples):
            samples[i] = self.brownian_motion.sample_path(
                end_time=1, n_steps=self.num_steps, initial_point=gs.array([0.0, 0.0])
            )
        return samples

    def test_normal_distribution_of_increments(self):
        r"""
        Test if the increments of Brownian motion are normally distributed.
        The null hypothesis is that the increments are normally distributed.
        """
        increments = self.samples[:, 1:] - self.samples[:, :-1]
        increments_flat = increments.flatten()
        statistic, p_value = normaltest(increments_flat)
        assert p_value > 0.05

    def test_independence_of_increments(self):
        r"""
        Test if the increments of Brownian motion are independent.
        The null hypothesis is that the increments are independent.
        """
        increments = self.samples[:, 1:] - self.samples[:, :-1]
        independent = True
        for i in range(1, increments.shape[1]):
            corr, p_value = pearsonr(
                increments[:, i - 1].flatten(), increments[:, i].flatten()
            )
            if abs(corr) > 0.5:
                independent = False
                break
        assert independent

    def test_normal_distribution_of_final_position(self):
        r"""
        Test if the final position of Brownian motion is normally distributed.
        The null hypothesis is that the final position is normally distributed.
        """
        final_positions = self.samples[:, -1]
        statistic, p_value = normaltest(final_positions.flatten())
        assert p_value > 0.05
