"""
Unit tests for landmarks space.
"""

import geomstats
import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry import hyperbolic_space
from geomstats.geometry import hypersphere
from geomstats.learning.k_means import KMeans


class TestLearningKMeans(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        pass

    def test_hyperbolic_space_k_means(self):
        x = gs.rand(100, 2)
        manifold = hyperbolic_space.HyperbolicSpace(2)
        metric = hyperbolic_space.HyperbolicMetric(2)
        x = manifold.intrinsic_to_extrinsic_coords(x)
        k_means = KMeans(2, metric, tol=1e-4)
        # test learning
        k_means.fit(x, max_iter=100)
        # test predict
        k_means.predict(x)

    def test_hypersphere_k_means(self):
        gs.random.seed(1234)

        manifold = hypersphere.Hypersphere(2)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=50)
        metric = hypersphere.HypersphereMetric(2)
        k_means = KMeans(5, metric, tol=1e-4)
        k_means.fit(x, max_iter=100)
        result = k_means.predict(x)
        expected = clustering.labels_[0]
        self.assertAllClose(expected, result)

if __name__ == '__main__':
    geomstats.tests.main()
