"""
Unit tests for landmarks space.
"""

import geomstats
import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry import hypersphere
from geomstats.learning.k_means import RiemannianKMeans


class TestLearningKMeans(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def test_hypersphere_k_means_fit(self):
        gs.random.seed(55)

        manifold = hypersphere.Hypersphere(2)
        metric = hypersphere.HypersphereMetric(2)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        k_means = RiemannianKMeans(metric, 1, tol=1e-3)
        k_means.fit(x)
        center = k_means.centroids
        mean = metric.mean(x)
        result = metric.dist(center, mean)
        expected = 0.
        self.assertAllClose(expected, result, atol=1e-2)

    def test_hypersphere_k_means_predict(self):
        gs.random.seed(1234)

        manifold = hypersphere.Hypersphere(2)
        metric = hypersphere.HypersphereMetric(2)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        k_means = RiemannianKMeans(metric, 5, tol=1e-5)
        k_means.fit(x, max_iter=100)
        result = k_means.predict(x)

        centroids = k_means.centroids
        expected = gs.array([int(metric.closest_neighbor_index(x_i, centroids))
                             for x_i in x])
        self.assertAllClose(expected, result)


if __name__ == '__main__':
    geomstats.tests.main()
