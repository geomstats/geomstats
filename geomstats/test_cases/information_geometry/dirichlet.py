import pytest

from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


class DirichletDistributionsTestCase(InformationManifoldMixinTestCase, OpenSetTestCase):
    pass


class DirichletMetricTestCase(RiemannianMetricTestCase):
    def test_jacobian_christoffels(self, base_point, expected, atol):
        res = self.space.metric.jacobian_christoffels(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_jacobian_christoffels_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.jacobian_christoffels(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)
