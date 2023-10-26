import pytest

import geomstats.backend as gs
from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class TangentPCATestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_fit_inverse_transform(self, n_samples, atol):
        X = expected = self.data_generator.random_point(n_samples)

        tangent_projected_data = self.estimator.fit_transform(X)
        res = self.estimator.inverse_transform(tangent_projected_data)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_fit_transform_and_transform_after_fit(self, n_samples, atol):
        X = self.data_generator.random_point(n_samples)

        res = self.estimator.fit_transform(X)
        res_ = self.estimator.fit(X).transform(X)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_n_components(self, n_samples):
        X = self.data_generator.random_point(n_samples)

        n_components_0 = self.estimator.n_components

        n_components = 2
        self.estimator.n_components = n_components

        self.estimator.fit(X)
        self.assertEqual(self.estimator.n_components_, n_components)

        self.estimator.n_components = n_components_0

    @pytest.mark.random
    def test_n_components_explained_variance_ratio(self, n_samples, atol):
        X = self.data_generator.random_point(n_samples)

        n_components_0 = self.estimator.n_components

        target = 0.9
        self.estimator.n_components = target

        self.estimator.fit(X)
        res = gs.cumsum(self.estimator.explained_variance_ratio_)[-1]
        self.assertTrue(res > target - atol)

        self.estimator.n_components = n_components_0

    @pytest.mark.random
    def test_n_components_mle(self, n_samples):
        if self.estimator.space.point_ndim > 1:
            return

        X = self.data_generator.random_point(n_samples)

        n_components_0 = self.estimator.n_components

        n_components = "mle"
        self.estimator.n_components = n_components

        self.estimator.fit(X)
        self.assertEqual(self.estimator.n_features_, gs.shape(X)[1])

        self.estimator.n_components = n_components_0
