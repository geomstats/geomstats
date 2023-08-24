"""Unit tests for Riemannian KMedoids."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.kmedoids import RiemannianKMedoids


class TestRiemannianKMedoids(tests.conftest.TestCase):
    _multiprocess_can_split_ = True

    def test_hypersphere_kmedoids_fit(self):
        gs.random.seed(55)

        manifold = Hypersphere(2)

        data = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmedoids = RiemannianKMedoids(manifold, n_clusters=1)
        center = kmedoids.fit(data).centroids_

        self.assertTrue(manifold.belongs(center))

    def test_hypersphere_kmedoids_predict(self):
        gs.random.seed(1234)
        dim = 2

        manifold = Hypersphere(dim)

        data = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmedoids = RiemannianKMedoids(manifold, n_clusters=5, max_iter=100)
        centroids = kmedoids.fit(data).centroids_
        result = kmedoids.predict(data)

        expected = gs.array(
            [
                int(manifold.metric.closest_neighbor_index(x_i, centroids))
                for x_i in data
            ]
        )
        self.assertAllClose(expected, result)
