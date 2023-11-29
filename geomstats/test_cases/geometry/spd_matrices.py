import pytest

import geomstats.backend as gs
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
)
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class SPDMatricesTestCaseMixins:
    def test_differential_power(self, power, tangent_vec, base_point, expected, atol):
        res = self.space.differential_power(power, tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_differential_power_vec(self, n_reps, power, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.differential_power(power, tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    power=power,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_inverse_differential_power(
        self, power, tangent_vec, base_point, expected, atol
    ):
        res = self.space.inverse_differential_power(power, tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inverse_differential_power_vec(self, n_reps, power, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.inverse_differential_power(power, tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    power=power,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_differential_log(self, tangent_vec, base_point, expected, atol):
        res = self.space.differential_log(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_inverse_differential_log(self, tangent_vec, base_point, expected, atol):
        res = self.space.inverse_differential_log(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_differential_exp(self, tangent_vec, base_point, expected, atol):
        res = self.space.differential_exp(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_inverse_differential_exp(self, tangent_vec, base_point, expected, atol):
        res = self.space.inverse_differential_exp(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_logm(self, mat, expected, atol):
        res = self.space.logm(mat)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_expm_after_logm(self, n_points, atol):
        mat = self.data_generator.random_point(n_points)

        logm = self.space.logm(mat)
        mat_ = self.space.expm(logm)

        self.assertAllClose(mat_, mat, atol=atol)

    @pytest.mark.random
    def test_logm_after_expm(self, n_points, atol):
        mat = self.data_generator.random_point(n_points)

        expm = self.space.expm(mat)
        mat_ = self.space.logm(expm)

        self.assertAllClose(mat_, mat, atol=atol)

    def test_cholesky_factor(self, mat, expected, atol):
        res = self.space.cholesky_factor(mat)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_cholesky_factor_belongs_to_positive_lower_triangular_matrices(
        self, n_points, atol
    ):
        mat = self.data_generator.random_point(n_points)

        cf = self.space.cholesky_factor(mat)
        res = PositiveLowerTriangularMatrices(self.space.n).belongs(cf, atol=atol)

        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    def test_differential_cholesky_factor(
        self, tangent_vec, base_point, expected, atol
    ):
        res = self.space.differential_cholesky_factor(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_differential_cholesky_factor_belongs_to_positive_lower_triangular_matrices(
        self, n_points, atol
    ):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        differential_cf = self.space.differential_cholesky_factor(
            tangent_vec, base_point
        )
        res = LowerTriangularMatrices(self.space.n).belongs(differential_cf, atol=atol)

        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)


class SPDMatricesTestCase(SPDMatricesTestCaseMixins, VectorSpaceOpenSetTestCase):
    pass


class SPDEuclideanMetricTestCase(RiemannianMetricTestCase):
    def test_exp_domain(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.exp_domain(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)
