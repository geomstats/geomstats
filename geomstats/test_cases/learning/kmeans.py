import pytest

from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class AgainstFrechetMeanTestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_against_frechet_mean(self, n_points, atol):
        if self.estimator.n_clusters > 1:
            raise ValueError("Test only works for one cluster.")
        X = self.data_generator.random_point(n_points=n_points)

        res = self.estimator.fit(X).cluster_centers_[0]
        res_ = self.other_estimator.fit(X).estimate_
        self.assertAllClose(res, res_, atol=atol)
