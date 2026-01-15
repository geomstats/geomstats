import random

import pytest

import geomstats.backend as gs
from geomstats.learning.mds import pairwise_dists
from geomstats.test_cases.learning._base import BaseEstimatorTestCase, TestCase
from geomstats.vectorization import repeat_point


class PairwiseDistsTestCase(TestCase):
    @pytest.mark.random
    def test_dists_among_selves(self, n_points, atol):
        points = repeat_point(
            self.space.random_point(n_samples=1), n_points, expand=True
        )
        print("F", n_points, points)
        pairwise_dist_matrix = pairwise_dists(self.space, points)
        self.assertAllClose(
            pairwise_dist_matrix, gs.zeros((n_points, n_points)), atol=atol
        )

    @pytest.mark.random
    def test_one_point(self, atol):
        self.test_dists_among_selves(n_samples=1, atol=atol)

    @pytest.mark.random
    def test_symmetric(self, n_points, atol):
        points = self.space.random_point(n_samples=n_points)
        pairwise_dist_matrix = pairwise_dists(self.space, points)
        self.assertAllClose(pairwise_dist_matrix, pairwise_dist_matrix.T, atol=atol)

    @pytest.mark.random
    def test_general(self, n_points, atol):
        points = self.space.random_point(n_samples=n_points)
        pairwise_dist_matrix = pairwise_dists(self.space, points)

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
    def test_minimal_fit(self):
        n_points = random.randint(3, 5)
        n_components = self.estimator.n_components

        X = self.data_generator.random_point(n_points=n_points)
        self.estimator.fit(X)
        embeddings = self.estimator.embedding_
        self.assertAllEqual(embeddings.shape, (n_points, n_components))

    @pytest.mark.random
    def test_minimal_fit_transform(self):
        n_points = random.randint(3, 5)
        n_components = self.estimator.n_components

        X = self.data_generator.random_point(n_points=n_points)
        embeddings = self.estimator.fit_transform(X)
        self.assertAllEqual(embeddings.shape, (n_points, n_components))
