"""Unit tests for Riemannian KMeans."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.kmeans import RiemannianKMeans


@tests.conftest.np_and_autograd_only
class TestRiemannianKMeans(tests.conftest.TestCase):
    _multiprocess_can_split_ = True

    def test_hypersphere_kmeans_fit(self):
        gs.random.seed(55)

        manifold = Hypersphere(2)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmeans = RiemannianKMeans(manifold, n_clusters=1, tol=1e-3)
        kmeans.fit(x)
        center = kmeans.centroids_

        mean = FrechetMean(manifold)
        mean.fit(x)

        result = manifold.metric.dist(center, mean.estimate_)
        expected = 0.0
        self.assertAllClose(expected, result)

    def test_spd_kmeans_fit(self):
        gs.random.seed(0)
        dim = 3
        n_points = 2
        space = SPDMatrices(dim)
        data = space.random_point(n_samples=n_points)

        kmeans = RiemannianKMeans(space, n_clusters=1)
        kmeans.fit(data)
        result = kmeans.centroids_[0]

        mean = FrechetMean(space).set(max_iter=100)
        mean.fit(data)
        expected = mean.estimate_
        self.assertAllClose(result, expected)

    def test_hypersphere_kmeans_predict(self):
        gs.random.seed(1234)
        dim = 2

        manifold = Hypersphere(dim)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmeans = RiemannianKMeans(manifold, 5, tol=1e-5)
        kmeans.fit(x)
        result = kmeans.predict(x)

        centroids = kmeans.centroids_
        expected = gs.array(
            [int(manifold.metric.closest_neighbor_index(x_i, centroids)) for x_i in x]
        )
        self.assertAllClose(expected, result)

    def _test_hypersphere_kmeans_init(
        self, init, *, n_features=4, n_clusters=3, seed=1
    ):
        gs.random.seed(seed)

        manifold = Hypersphere(n_features - 1)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmeans = RiemannianKMeans(manifold, n_clusters, tol=1e-3, init=init)
        kmeans.fit(x)

        centroids = kmeans.centroids_
        result = centroids.shape
        expected = (n_clusters, n_features)
        self.assertAllClose(expected, result)

    def test_hypersphere_kmeans_init_kmeanspp(self):
        self._test_hypersphere_kmeans_init("kmeans++")

    def test_hypersphere_kmeans_init_array(self):
        n_features = 4
        n_clusters = 3

        manifold = Hypersphere(n_features - 1)
        centroids = manifold.random_von_mises_fisher(kappa=10, n_samples=n_clusters)

        self._test_hypersphere_kmeans_init(
            centroids, n_features=n_features, n_clusters=n_clusters
        )

    def test_hypersphere_kmeans_init_callable(self):
        # Note that _test_hypersphere_kmeans_init sets the random seed before
        # make_centroids is called by RiemannianKMeans.fit.
        def make_centroids(X, n_clusters):
            n_features = X.shape[1]
            manifold = Hypersphere(n_features - 1)
            centroids = manifold.random_von_mises_fisher(kappa=10, n_samples=n_clusters)
            return centroids

        self._test_hypersphere_kmeans_init(make_centroids)
