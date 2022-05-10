"""Unit tests for Riemannian Mean Shift method."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.riemannian_mean_shift import (
    RiemannianMeanShift as riemannian_mean_shift,
)


class TestRiemannianMeanShift(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    @geomstats.tests.np_autograd_and_torch_only
    def test_hypersphere_riemannian_mean_shift_predict(self):
        gs.random.seed(1234)
        dim = 2

        manifold = Hypersphere(dim)
        metric = HypersphereMetric(dim)
        cluster = manifold.random_von_mises_fisher(kappa=100, n_samples=10)

        rms = riemannian_mean_shift(
            manifold=manifold,
            metric=metric,
            bandwidth=0.6,
            tol=1e-4,
            n_centers=2,
            max_iter=100,
        )
        rms.fit(cluster)
        result = rms.predict(cluster)

        closest_centers = []
        for point in cluster:
            closest_center = metric.closest_neighbor_index(point, rms.centers)
            closest_centers.append(rms.centers[closest_center, :])
        expected = gs.array(closest_centers)

        self.assertAllClose(expected, result)

    @geomstats.tests.np_autograd_and_torch_only
    def test_single_cluster_riemannian_mean_shift(self):
        gs.random.seed(10)

        sphere = Hypersphere(dim=2)
        metric = HypersphereMetric(2)

        cluster = sphere.random_von_mises_fisher(kappa=100, n_samples=10)

        rms = riemannian_mean_shift(
            manifold=sphere,
            metric=metric,
            bandwidth=float("inf"),
            tol=1e-4,
            n_centers=1,
            max_iter=1,
        )
        rms.fit(cluster)
        center = rms.predict(cluster)

        mean = FrechetMean(metric=metric, init_step_size=1.0)
        mean.fit(cluster)

        result = center[0]
        expected = mean.estimate_

        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_autograd_only
    def test_double_cluster_riemannian_mean_shift(self):
        gs.random.seed(10)
        number_of_samples = 20
        sphere = Hypersphere(dim=2)
        metric = HypersphereMetric(2)

        cluster = sphere.random_von_mises_fisher(kappa=20, n_samples=number_of_samples)

        special_orthogonal = SpecialOrthogonal(3)
        rotation1 = special_orthogonal.random_uniform()
        rotation2 = special_orthogonal.random_uniform()

        cluster_1 = cluster @ rotation1
        cluster_2 = cluster @ rotation2

        combined_cluster = gs.concatenate((cluster_1, cluster_2))
        rms = riemannian_mean_shift(
            manifold=sphere, metric=metric, bandwidth=0.3, tol=1e-4, n_centers=2
        )

        rms.fit(combined_cluster)
        closest_centers = rms.predict(combined_cluster)

        count_in_first_cluster = 0
        for point in closest_centers:
            if gs.allclose(point, rms.centers[0]):
                count_in_first_cluster += 1

        count_in_second_cluster = 0
        for point in closest_centers:
            if gs.allclose(point, rms.centers[1]):
                count_in_second_cluster += 1

        self.assertEqual(
            combined_cluster.shape[0], count_in_first_cluster + count_in_second_cluster
        )
