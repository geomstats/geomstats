import pytest

import geomstats.backend as gs
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.geometry.base import (
    FiberBundleTestCase,
    OpenSetTestCase,
    RiemannianMetricTestCase,
)
from geomstats.test.geometry.general_linear import GeneralLinearTestCase
from geomstats.test.vectorization import generate_vectorization_data


class BuresWassersteinBundle(FiberBundle, GeneralLinear):
    def __init__(self, n):
        super().__init__(
            n=n,
            group=SpecialOrthogonal(n),
        )

    @staticmethod
    def default_metric():
        return MatricesMetric

    @staticmethod
    def riemannian_submersion(point):
        return Matrices.mul(point, Matrices.transpose(point))

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        product = Matrices.mul(base_point, Matrices.transpose(tangent_vec))
        return 2 * Matrices.to_symmetric(product)

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        if base_point is None:
            if fiber_point is not None:
                base_point = self.riemannian_submersion(fiber_point)
            else:
                raise ValueError(
                    "Either a point (of the total space) or a "
                    "base point (of the base manifold) must be "
                    "given."
                )
        sylvester = gs.linalg.solve_sylvester(base_point, base_point, tangent_vec)
        return Matrices.mul(sylvester, fiber_point)

    @staticmethod
    def lift(point):
        return gs.linalg.cholesky(point)


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


class BuresWassersteinBundleTestCase(FiberBundleTestCase, GeneralLinearTestCase):
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


class SPDLogEuclideanMetricTestCase(RiemannianMetricTestCase):
    pass
