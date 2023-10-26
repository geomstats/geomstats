import pytest

import geomstats.backend as gs
from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class ClusterInitializationTestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_initialization_belongs(self, n_samples, atol):
        X = self.data_generator.random_point(n_points=n_samples)
        init_cluster_centers = self.estimator._pick_init_cluster_centers(X)
        belongs = self.estimator.space.belongs(init_cluster_centers, atol=atol)
        self.assertTrue(gs.all(belongs))


class AgainstFrechetMeanTestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_against_frechet_mean(self, n_samples, atol):
        if self.estimator.n_clusters > 1:
            raise ValueError("Test only works for one cluster.")
        X = self.data_generator.random_point(n_points=n_samples)

        res = self.estimator.fit(X).cluster_centers_[0]
        res_ = self.other_estimator.fit(X).estimate_
        self.assertAllClose(res, res_, atol=atol)
