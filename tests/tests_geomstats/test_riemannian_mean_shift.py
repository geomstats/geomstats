""" Unit tests for Riemannian Mean Shift method """

import numpy as np

import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.riemannian_mean_shift import RiemannianMeanShift as RMS


@geomstats.tests.np_and_autograd_only
class TestRiemannianMeanShift(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def test_hypersphere_riemannian_mean_shift_fit(self):
        np.random.seed(10)

        sphere = Hypersphere(dim=2)
        metric = HypersphereMetric(2)

        cluster = sphere.random_von_mises_fisher(kappa=100, n_samples=10)

        rms = RMS(
            manifold=sphere,
            metric=metric,
            bandwidth=float("inf"),
            tol=1e-4,
            n_centers=1,
            max_iter=1,
        )
        rms.fit(cluster)
        center = rms.centers

        mean = FrechetMean(metric=metric, init_step_size=1.0)
        mean.fit(cluster)

        result = center[0]
        expected = mean.estimate_

        self.assertAllClose(expected, result)
