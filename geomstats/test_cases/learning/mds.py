import pytest

import geomstats.backend as gs
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
)
from geomstats.vectorization import repeat_point


class MDSTestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_dissimilarity_against_self(self, n_samples, atol):
        X = repeat_point(self.data_generator.random_point(n_points=1), n_samples)
        self.test_dissimilarity_matrix(X, gs.zeros((n_samples, n_samples)), atol=atol)

    def test_dissimilarity_matrix(self, X, expected, atol):
        dissim_matrix = self.estimator.fit(X).dissimilarity_matrix_
        self.assertAllClose(dissim_matrix, expected, atol=atol)

    def test_dissimilarity_matrix2(self, X, expected, atol):
        dissim_matrix = self.estimator.fit(X).dissimilarity_matrix_
        self.assertAllClose(dissim_matrix, expected, atol=atol)

    # test distances preserved in embedding?
