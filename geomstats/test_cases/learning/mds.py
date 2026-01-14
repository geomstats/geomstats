import random

import pytest

import geomstats.backend as gs
from geomstats.learning.mds import pairwise_dists
from geomstats.test_cases.learning._base import BaseEstimatorTestCase
from geomstats.vectorization import repeat_point


class PairwiseDistsTestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_dists_among_selves(self, n_samples, atol):
        points = repeat_point(
            self.data_generator.random_point(n_points=1), n_samples, expand=True
        )
        pairwise_dist_matrix = pairwise_dists(self.estimator.space, points)
        self.assertAllClose(
            pairwise_dist_matrix, gs.zeros((n_samples, n_samples)), atol=atol
        )

    @pytest.mark.random
    def test_one_point(self, n_samples, atol):
        self.test_dists_among_selves(n_samples=1, atol=atol)


class EyePairwiseDistsTestCase(BaseEstimatorTestCase):
    def test_euclidean_eye(self, points, expected, atol):
        pairwise_dist_matrix = pairwise_dists(self.estimator.space, points)
        self.assertAllClose(pairwise_dist_matrix, expected, atol=atol)


class MDSTestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_minimal_fit(self, atol):
        n_points = random.randint(3, 5)
        n_components = self.estimator.n_components

        X = self.data_generator.random_point(n_points=n_points)
        self.estimator.fit(X)
        embeddings = self.estimator.embedding_
        self.assertAllEqual(embeddings.shape, (n_points, n_components))

    @pytest.mark.random
    def test_minimal_fit_transform(self, atol):
        n_points = random.randint(3, 5)
        n_components = self.estimator.n_components

        X = self.data_generator.random_point(n_points=n_points)
        embeddings = self.estimator.fit_transform(X)
        self.assertAllEqual(embeddings.shape, (n_points, n_components))
