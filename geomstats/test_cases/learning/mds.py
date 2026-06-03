import random

import pytest

import geomstats.backend as gs
from geomstats.learning.mds import pairwise_dists
from geomstats.test_cases.learning._base import BaseEstimatorTestCase, TestCase
from geomstats.vectorization import repeat_point


class PairwiseDistsTestCase(TestCase):
    def test_dists(self, points, expected, atol):
        pairwise_dist_matrix = pairwise_dists(points, self.space.metric.dist)
        self.assertAllClose(
            pairwise_dist_matrix,
            expected,
            atol=atol,
        )

    @pytest.mark.random
    def test_dists_among_selves(self, n_points, atol):
        points = repeat_point(
            self.space.random_point(n_samples=1), n_points, expand=True
        )
        self.test_dists(points, gs.zeros((n_points, n_points)), atol=atol)

    @pytest.mark.random
    def test_symmetric(self, n_points, atol):
        points = self.space.random_point(n_samples=n_points)
        pairwise_dist_matrix = pairwise_dists(points, self.space.metric.dist)
        self.assertAllClose(pairwise_dist_matrix, pairwise_dist_matrix.T, atol=atol)

    @pytest.mark.random
    def test_matrix_indices(self, n_points, atol):
        points = self.space.random_point(n_samples=n_points)
        pairwise_dist_matrix = pairwise_dists(points, self.space.metric.dist)

        rand_i, rand_j = (
            random.randint(0, n_points - 1),
            random.randint(0, n_points - 1),
        )
        self.assertAllClose(
            [pairwise_dist_matrix[rand_i, rand_j]],
            [self.space.metric.dist(points[rand_i], points[rand_j])],
            atol=atol,
        )


class MDSTestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_fit_runs(self, n_samples):
        X = self.data_generator.random_point(n_points=n_samples)

        self.estimator.fit(X)
        embeddings = self.estimator.embedding_
        n_components = self.estimator.n_components

        self.assertAllEqual(embeddings.shape, (n_samples, n_components))

    @pytest.mark.random
    def test_fit_transform_runs(self, n_samples):
        X = self.data_generator.random_point(n_points=n_samples)

        embeddings = self.estimator.fit_transform(X)
        n_components = self.estimator.n_components

        self.assertAllEqual(embeddings.shape, (n_samples, n_components))
