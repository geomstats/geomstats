import pytest

import geomstats.backend as gs
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


def scalar_curvature(base_point):
    kappa = base_point[..., 0]
    return (gs.polygamma(1, kappa) + kappa * gs.polygamma(2, kappa)) / (
        2 * (-1 + kappa * gs.polygamma(1, kappa)) ** 2
    )


class GammaDistributionsTestCase(
    InformationManifoldMixinTestCase, VectorSpaceOpenSetTestCase
):
    def _check_sample_belongs_to_support(self, sample, atol):
        self.assertTrue(gs.all(sample > 0.0))

    def test_natural_to_standard(self, point, expected, atol):
        res = self.space.natural_to_standard(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_standard_to_natural(self, point, expected, atol):
        res = self.space.standard_to_natural(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_standard_to_natural_vec(self, n_reps, atol):
        point = self.data_generator.random_point_standard()

        expected = self.space.standard_to_natural(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_standard_to_natural_after_natural_to_standard(self, n_points, atol):
        point = self.data_generator.random_point(n_points)

        point_standard = self.space.natural_to_standard(point)
        point_ = self.space.standard_to_natural(point_standard)

        self.assertAllClose(point_, point, atol=atol)

    def test_tangent_natural_to_standard(self, vec, base_point, expected, atol):
        res = self.space.tangent_natural_to_standard(vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_tangent_standard_to_natural(self, vec, base_point, expected, atol):
        res = self.space.tangent_standard_to_natural(vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tangent_standard_to_natural_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point_standard()
        vec = self.data_generator.random_tangent_vec_standard(base_point)

        expected = self.space.tangent_standard_to_natural(vec, base_point)

        vec_data = generate_vectorization_data(
            data=[dict(vec=vec, base_point=base_point, expected=expected, atol=atol)],
            arg_names=["vec", "base_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_tangent_standard_to_natural_after_tangent_natural_to_standard(
        self, n_points, atol
    ):
        base_point = self.data_generator.random_point(n_points)
        vec = self.data_generator.random_tangent_vec(base_point)

        vec_standard = self.space.tangent_natural_to_standard(vec, base_point)
        base_point_standard = self.space.natural_to_standard(base_point)

        vec_ = self.space.tangent_standard_to_natural(vec_standard, base_point_standard)
        self.assertAllClose(vec_, vec, atol=atol)

    def test_maximum_likelihood_fit(self, data, expected, atol):
        res = self.space.maximum_likelihood_fit(data)
        self.assertAllClose(res, expected, atol=atol)


class GammaMetricTestCase(RiemannianMetricTestCase):
    def test_jacobian_christoffels(self, base_point, expected, atol):
        res = self.space.metric.jacobian_christoffels(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_scalar_curvature_against_closed_form(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.scalar_curvature(base_point)
        expected = scalar_curvature(base_point)
        self.assertAllClose(res, expected, atol=atol)
