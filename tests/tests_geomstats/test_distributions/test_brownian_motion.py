import random

import pytest
from scipy.stats import normaltest, pearsonr

import geomstats.backend as gs
from geomstats.distributions.brownian_motion import BrownianMotion
from geomstats.geometry.euclidean import Euclidean
from geomstats.test.test_case import TestCase, autograd_and_torch_only


@autograd_and_torch_only
@pytest.mark.slow
class TestBrownianMotion(TestCase):
    """Verify the axioms of Brownian Motion"""

    def setup_method(self):
        self.space = Euclidean(dim=random.randint(2, 3))
        self.brownian_motion = BrownianMotion(self.space)
        self.samples = self._generate_samples(
            self.space, n_samples=100, n_steps=random.randint(30, 50)
        )

    def _generate_samples(
        self,
        space,
        n_samples,
        n_steps,
    ):
        """Generate samples of Brownian motion."""
        initial_point = space.random_point()
        samples = []
        for _ in range(n_samples):
            samples.append(
                self.brownian_motion.sample_path(
                    end_time=1.0,
                    n_steps=n_steps,
                    initial_point=initial_point,
                )
            )
        return gs.stack(samples)

    def test_normal_distribution_of_increments(self):
        """Test if the increments of Brownian motion are normally distributed.

        The null hypothesis is that the increments are normally distributed.
        """
        increments = self.samples[:, 1:] - self.samples[:, :-1]
        increments_flat = increments.flatten()
        statistic, p_value = normaltest(increments_flat)
        self.assertTrue(p_value > 0.05)

    def test_independence_of_increments(self):
        """Test if the increments of Brownian motion are independent.

        The null hypothesis is that the increments are independent.
        """
        increments = self.samples[:, 1:] - self.samples[:, :-1]
        for i in range(1, increments.shape[1]):
            corr, p_value = pearsonr(
                increments[:, i - 1].flatten(), increments[:, i].flatten()
            )
            if abs(corr) > 0.5:
                raise ValueError(
                    "Increments are not independent", msg=f"p-value: {p_value}"
                )

    @pytest.mark.xfail
    def test_normal_distribution_of_final_position(self):
        r"""Test if the final position of Brownian motion is normally distributed.

        The null hypothesis is that the final position is normally distributed.
        """
        final_positions = self.samples[:, -1]
        statistic, p_value = normaltest(final_positions.flatten())
        self.assertTrue(p_value > 0.05, msg=f"p-value: {p_value}")
