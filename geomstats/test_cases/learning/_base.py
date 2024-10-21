import pytest

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.vectorization import repeat_point


class BaseEstimatorTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.estimator.space)


class MeanEstimatorMixinsTestCase:
    def _self_assert_same_point(self, point, point_, atol):
        self.assertAllClose(point, point_, atol)

    def test_fit(self, X, expected, atol, weights=None):
        kwargs = {} if weights is None else {"weights": weights}
        res = self.estimator.fit(X, **kwargs).estimate_
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_one_point(self, atol):
        X = gs.expand_dims(self.data_generator.random_point(n_points=1), axis=0)

        mean = self.estimator.fit(X).estimate_
        self._self_assert_same_point(mean, X[0], atol)

    @pytest.mark.random
    def test_n_times_same_point(self, n_samples, atol):
        X = repeat_point(self.data_generator.random_point(n_points=1), n_samples)

        mean = self.estimator.fit(X).estimate_
        self._self_assert_same_point(mean, X[0], atol)

    @pytest.mark.random
    def test_estimate_belongs(self, n_samples, atol):
        X = self.data_generator.random_point(n_points=n_samples)
        mean = self.estimator.fit(X).estimate_
        belongs = self.estimator.space.belongs(mean, atol=atol)
        self.assertTrue(belongs)


class ClusterMixinsTestCase:
    @pytest.mark.random
    def test_n_repeated_clusters(self, n_samples, atol):
        n_clusters = self.estimator.n_clusters
        X = []
        for _ in range(n_clusters):
            X.extend(
                repeat_point(self.data_generator.random_point(n_points=1), n_samples)
            )
        X = gs.array(X)

        cluster_centers = self.estimator.fit(X).cluster_centers_
        labels = self.estimator.predict(X)

        for k in range(n_clusters):
            i = n_samples * k
            j = i + n_samples
            X_ = X[i:j]
            labels_ = labels[i:j]

            self.assertTrue(gs.unique(labels_).shape[0], 1)
            cluster_center = cluster_centers[labels_[0]]

            dist = self.estimator.space.metric.dist(X_, cluster_center)
            self.assertAllClose(dist, gs.zeros(n_samples), atol=atol)

    @pytest.mark.random
    def test_cluster_assignment(self, n_samples):
        X = self.data_generator.random_point(n_samples)
        space = self.estimator.space

        cluster_centers = self.estimator.fit(X).cluster_centers_
        result = self.estimator.predict(X)

        expected = gs.array(
            [
                int(space.metric.closest_neighbor_index(x_i, cluster_centers))
                for x_i in X
            ]
        )
        self.assertAllEqual(result, expected)

    @pytest.mark.random
    def test_cluster_centers_belong(self, n_samples):
        X = self.data_generator.random_point(n_samples)
        space = self.estimator.space

        cluster_centers = self.estimator.fit(X).cluster_centers_

        result = space.belongs(cluster_centers)
        expected = gs.ones((self.estimator.n_clusters,), dtype=bool)
        self.assertAllEqual(result, expected)

    @pytest.mark.shape
    def test_cluster_centers_shape(self, n_samples):
        X = self.data_generator.random_point(n_samples)
        space = self.estimator.space

        cluster_centers = self.estimator.fit(X).cluster_centers_

        n_clusters = self.estimator.n_clusters

        self.assertAllEqual(gs.shape(cluster_centers), (n_clusters, *space.shape))
