"""Unit tests for Riemannian Mean Shift method."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.riemannian_mean_shift import (
    RiemannianMeanShift as riemannian_mean_shift,
)


class TestRiemannianMeanShift(tests.conftest.TestCase):
    _multiprocess_can_split_ = True

    @tests.conftest.np_autograd_and_torch_only
    def test_hypersphere_predict(self):
        gs.random.seed(1234)

        sphere = Hypersphere(dim=2)
        metric = sphere.metric
        cluster = sphere.random_von_mises_fisher(kappa=100, n_samples=10)

        rms = riemannian_mean_shift(
            manifold=sphere,
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

    @tests.conftest.np_autograd_and_torch_only
    def test_single_cluster(self):
        gs.random.seed(10)

        sphere = Hypersphere(dim=2)
        metric = sphere.metric

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

    @staticmethod
    def _init_double_cluster(
        seed=10,
        num_of_samples=20,
        size_of_dim=2,
        kappa_value=20,
        orthogonality_of_sphere=3,
        bandwidth=0.3,
        tol=1e-4,
        num_of_centers=2,
    ):
        gs.random.seed(seed)
        number_of_samples = num_of_samples
        sphere = Hypersphere(size_of_dim)
        metric = sphere.metric

        cluster = sphere.random_von_mises_fisher(
            kappa=kappa_value, n_samples=number_of_samples
        )

        special_orthogonal = SpecialOrthogonal(orthogonality_of_sphere)
        rotation1 = special_orthogonal.random_uniform()
        rotation2 = special_orthogonal.random_uniform()

        cluster_1 = cluster @ rotation1
        cluster_2 = cluster @ rotation2

        combined_cluster = gs.concatenate((cluster_1, cluster_2))
        rms = riemannian_mean_shift(
            manifold=sphere,
            metric=metric,
            bandwidth=bandwidth,
            tol=tol,
            n_centers=num_of_centers,
        )

        rms.fit(combined_cluster)

        return combined_cluster, rms

    @tests.conftest.np_and_autograd_only
    def test_double_cluster(self):
        combined_cluster, rms = self._init_double_cluster()
        closest_centers = rms.predict(combined_cluster)

        count_in_first_cluster = 0
        count_in_second_cluster = 0

        for point in closest_centers:
            if gs.allclose(point, rms.centers[0]):
                count_in_first_cluster += 1
            elif gs.allclose(point, rms.centers[1]):
                count_in_second_cluster += 1

        self.assertEqual(
            combined_cluster.shape[0], count_in_first_cluster + count_in_second_cluster
        )

    @tests.conftest.np_and_autograd_only
    def test_predict_labels(self):
        combined_cluster, rms = self._init_double_cluster()
        closest_center_labels = rms.predict_labels(combined_cluster)

        first_label_count = 0
        second_label_count = 0

        for label in closest_center_labels:
            if gs.allclose(label, 0):
                first_label_count += 1
            elif gs.allclose(label, 1):
                second_label_count += 1

        self.assertEqual(
            combined_cluster.shape[0], first_label_count + second_label_count
        )
