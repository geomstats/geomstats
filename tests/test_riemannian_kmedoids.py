"""Unit tests for Riemannian KMedoids."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry import hypersphere
from geomstats.learning.kmedoids import RiemannianKMedoids


class TestRiemannianKMedoids(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    @geomstats.tests.np_and_pytorch_only
    def test_hypersphere_kmedoids_fit(self):
        gs.random.seed(55)

        manifold = hypersphere.Hypersphere(2)
        metric = hypersphere.HypersphereMetric(2)

        data = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmedoids = RiemannianKMedoids(metric=metric, n_clusters=1)
        center = kmedoids.fit(data)

        self.assertTrue(manifold.belongs(center))

    @geomstats.tests.np_and_pytorch_only
    def test_hypersphere_kmedoids_predict(self):
        gs.random.seed(1234)
        dim = 2

        manifold = hypersphere.Hypersphere(dim)
        metric = hypersphere.HypersphereMetric(dim)

        data = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmedoids = RiemannianKMedoids(metric, n_clusters=5)
        centroids = kmedoids.fit(data, max_iter=100)
        result = kmedoids.predict(data)

        expected = gs.array(
            [int(metric.closest_neighbor_index(x_i, centroids))
             for x_i in data])
        self.assertAllClose(expected, result)
