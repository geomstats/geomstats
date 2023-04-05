import pytest

import geomstats.backend as gs
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
)
from geomstats.test.geometry.base import OpenSetTestCase, RiemannianMetricTestCase
from geomstats.test.vectorization import generate_vectorization_data


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

    @pytest.mark.vec
    def test_differential_log_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.differential_log(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
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

    def test_inverse_differential_log(self, tangent_vec, base_point, expected, atol):
        res = self.space.inverse_differential_log(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inverse_differential_log_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.inverse_differential_log(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
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

    def test_differential_exp(self, tangent_vec, base_point, expected, atol):
        res = self.space.differential_exp(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_differential_exp_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.differential_exp(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
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

    def test_inverse_differential_exp(self, tangent_vec, base_point, expected, atol):
        res = self.space.inverse_differential_exp(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inverse_differential_exp_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.inverse_differential_exp(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
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

    def test_logm(self, mat, expected, atol):
        res = self.space.logm(mat)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_logm_vec(self, n_reps, atol):
        mat = self.data_generator.random_point()
        expected = self.space.logm(mat)

        vec_data = generate_vectorization_data(
            data=[dict(mat=mat, expected=expected, atol=atol)],
            arg_names=[],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_cholesky_factor(self, mat, expected, atol):
        res = self.space.cholesky_factor(mat)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_cholesky_factor_vec(self, n_reps, atol):
        mat = self.data_generator.random_point()
        expected = self.space.cholesky_factor(mat)

        vec_data = generate_vectorization_data(
            data=[dict(mat=mat, expected=expected, atol=atol)],
            arg_names=[],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

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

    @pytest.mark.vec
    def test_differential_cholesky_factor_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.differential_cholesky_factor(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
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


class SPDMatricesTestCase(SPDMatricesTestCaseMixins, OpenSetTestCase):
    pass


class SPDAffineMetricTestCase(RiemannianMetricTestCase):
    pass


class SPDBuresWassersteinMetricTestCase(RiemannianMetricTestCase):
    pass


class SPDEuclideanMetricTestCase(RiemannianMetricTestCase):
    def test_exp_domain(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.exp_domain(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_exp_domain_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.exp_domain(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
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
