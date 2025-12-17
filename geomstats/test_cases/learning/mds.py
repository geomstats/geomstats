import pytest

from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
)


class MetricMDSTestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_dissimilarity_matrix(self, X, expected, atol):
        _ = self.estimator.fit_transform(X)
        dissim_matrix = self.estimator.dissimilarity_matrix_
        print(dissim_matrix)
        self.assertAllClose(dissim_matrix, expected, atol=atol)

    # test distances preserved in embedding?
