# TODO: test initialization

import pytest

import geomstats.backend as gs
from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class RiemannianKMeansTestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_one_cluster(self, n_points, atol):
        if self.estimator.n_clusters > 1:
            raise ValueError("Cannot run this test for n_clusters > 1.")

        X = self.data_generator.random_point(n_points)

        centroids = self.estimator.fit(X).centroids_
        mean = self.estimator.mean_estimator.fit(X).estimate_

        result = self.estimator.space.metric.dist(centroids[0], mean)
        self.assertAllClose(result, 0.0, atol=atol)

    @pytest.mark.random
    def test_cluster_assignment(self, n_points):
        X = self.data_generator.random_point(n_points)
        space = self.estimator.space

        centroids = self.estimator.fit(X).centroids_
        result = self.estimator.predict(X)

        expected = gs.array(
            [int(space.metric.closest_neighbor_index(x_i, centroids)) for x_i in X]
        )
        self.assertAllEqual(result, expected)

    @pytest.mark.random
    def test_centroids_belong(self, n_points):
        X = self.data_generator.random_point(n_points)
        space = self.estimator.space

        centroids = self.estimator.fit(X).centroids_

        result = space.belongs(centroids)
        expected = gs.ones((self.estimator.n_clusters,), dtype=bool)
        self.assertAllEqual(result, expected)

    @pytest.mark.shape
    @pytest.mark.random
    def test_centroids_shape(self, n_points):
        X = self.data_generator.random_point(n_points)
        space = self.estimator.space

        centroids = self.estimator.fit(X).centroids_

        n_clusters = self.estimator.n_clusters

        self.assertAllEqual(gs.shape(centroids), (n_clusters, *space.shape))
