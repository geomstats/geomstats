import pytest

import geomstats.backend as gs
from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class ToTangentSpaceTestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_fit_transform_is_tangent(self, n_points, atol):
        X = self.data_generator.random_point(n_points=n_points)

        tangent_vecs = self.estimator.fit_transform(X)
        res = self.estimator.space.is_tangent(
            tangent_vecs, self.estimator.base_point_, atol=atol
        )

        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_inverse_transform_after_transform(self, n_points, atol):
        X = self.data_generator.random_point(n_points=n_points)

        self.estimator.fit(X)
        tangent_vecs = self.estimator.transform(X)
        X_ = self.estimator.inverse_transform(tangent_vecs)

        self.assertAllClose(X_, X, atol=atol)
