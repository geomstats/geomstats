"""Unit tests for the manifold of symmetric positive definite matrices."""


import math
import random

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.spd_matrices import (
    SPDMatrices,
    SPDMetricAffine,
    SPDMetricBuresWasserstein,
    SPDMetricEuclidean,
    SPDMetricLogEuclidean,
)
from tests.conftest import Parametrizer, TestCase, TestData

SQRT_2 = math.sqrt(2.0)
LN_2 = math.log(2.0)
EXP_1 = math.exp(1.0)
EXP_2 = math.exp(2.0)
SINH_1 = math.sinh(1.0)


class TestSPDMatrices(TestCase, metaclass=Parametrizer):
    """Test of SPDMatrices methods."""

    class TestDataSPDMatrices(TestData):
        def belongs_data(self):
            smoke_data = [
                dict(n=2, mat=[[3.0, -1.0], [-1.0, 3.0]], expected=True),
                dict(n=2, mat=[[1.0, 1.0], [2.0, 1.0]], expected=False),
                dict(
                    n=3,
                    mat=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                    expected=False,
                ),
                dict(
                    n=2,
                    mat=[[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]],
                    expected=[True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def projection_data(self):
            smoke_data = [
                dict(
                    n=2, mat=[[1.0, 0.0], [0.0, 1.0]], expected=[[1.0, 0.0], [0.0, 1.0]]
                ),
                dict(
                    n=2,
                    mat=[[-1.0, 0.0], [0.0, -2.0]],
                    expected=[[gs.atol, 0.0], [0.0, gs.atol]],
                ),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_data(self):
            smoke_data = [
                dict(n=1, n_samples=1),
                dict(n=2, n_samples=1),
                dict(n=10, n_samples=10),
                dict(n=10, n_samples=1000),
            ]
            return self.generate_tests(smoke_data)

        def logm_data(self):
            smoke_data = [
                dict(
                    spd_mat=[[1.0, 0.0], [0.0, 1.0]], expected=[[0.0, 0.0], [0.0, 0.0]]
                )
            ]
            return self.generate_tests(smoke_data)

        def cholesky_factor_data(self):
            smoke_data = [
                dict(
                    n=2,
                    spd_mat=[[[1.0, 2.0], [2.0, 5.0]], [[1.0, 0.0], [0.0, 1.0]]],
                    expected=[[[1.0, 0.0], [2.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                ),
                dict(
                    n=3,
                    spd_mat=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
                    expected=[
                        [SQRT_2, 0.0, 0.0],
                        [0.0, SQRT_2, 0.0],
                        [0.0, 0.0, SQRT_2],
                    ],
                ),
            ]
            return self.generate_tests(smoke_data)

        def cholesky_factor_belongs_data(self):
            list_n = random.sample(range(1, 100), 10)
            n_samples = 10
            random_data = [
                dict(n=n, mat=SPDMatrices(n).random_point(n_samples)) for n in list_n
            ]
            return self.generate_tests([], random_data)

        def differential_cholesky_factor_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec=[[1.0, 1.0], [1.0, 1.0]],
                    base_point=[[4.0, 2.0], [2.0, 5.0]],
                    expected=[[1 / 4, 0.0], [3 / 8, 1 / 16]],
                )
            ]
            return self.generate_tests(smoke_data)

        def differential_power_data(self):
            smoke_data = [
                dict(
                    power=0.5,
                    tangent_vec=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                    base_point=[[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]],
                    expected=[
                        [1.0, 1 / 3, 1 / 3],
                        [1 / 3, 0.125, 0.125],
                        [1 / 3, 0.125, 0.125],
                    ],
                )
            ]
            return self.generate_tests(smoke_data)

        def inverse_differential_power_data(self):
            smoke_data = [
                dict(
                    power=0.5,
                    tangent_vec=[
                        [1.0, 1 / 3, 1 / 3],
                        [1 / 3, 0.125, 0.125],
                        [1 / 3, 0.125, 0.125],
                    ],
                    base_point=[[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]],
                    expected=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                )
            ]
            return self.generate_tests(smoke_data)

        def differential_log_data(self):
            smoke_data = [
                dict(
                    tangent_vec=[[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]],
                    base_point=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]],
                    expected=[
                        [1.0, 1.0, 2 * LN_2],
                        [1.0, 1.0, 2 * LN_2],
                        [2 * LN_2, 2 * LN_2, 1],
                    ],
                )
            ]
            return self.generate_tests(smoke_data)

        def inverse_differential_log_data(self):
            smoke_data = [
                dict(
                    tangent_vec=[
                        [1.0, 1.0, 2 * LN_2],
                        [1.0, 1.0, 2 * LN_2],
                        [2 * LN_2, 2 * LN_2, 1],
                    ],
                    base_point=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]],
                    expected=[[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]],
                )
            ]

            return self.generate_tests(smoke_data)

        def differential_exp_data(self):
            smoke_data = [
                dict(
                    tangent_vec=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    base_point=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                    expected=[
                        [EXP_1, EXP_1, SINH_1],
                        [EXP_1, EXP_1, SINH_1],
                        [SINH_1, SINH_1, 1 / EXP_1],
                    ],
                )
            ]
            return self.generate_tests(smoke_data)

        def inverse_differential_exp_data(self):
            smoke_data = [
                dict(
                    tangent_vec=[
                        [EXP_1, EXP_1, SINH_1],
                        [EXP_1, EXP_1, SINH_1],
                        [SINH_1, SINH_1, 1 / EXP_1],
                    ],
                    base_point=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                    expected=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                )
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataSPDMatrices()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(SPDMatrices(n).belongs(gs.array(mat)), gs.array(expected))

    def test_random_point_belongs(self, n, n_samples):
        space = SPDMatrices(n)
        self.assertAllClose(
            gs.all(space.belongs(space.random_point(n_samples))), gs.array(True)
        )

    def test_projection(self, n, mat, expected):
        self.assertAllClose(
            SPDMatrices(n).projection(gs.array(mat)), gs.array(expected)
        )

    def test_logm(self, spd_mat, logm):
        self.assertAllClose(SPDMatrices.logm(gs.array(spd_mat)), gs.array(logm))

    def test_cholesky_factor(self, n, spd_mat, cf):
        result = SPDMatrices.cholesky_factor(gs.array(spd_mat))

        self.assertAllClose(result, gs.array(cf))
        self.assertAllClose(
            gs.all(PositiveLowerTriangularMatrices(n).belongs(result)),
            gs.array(True),
        )

    def test_differential_cholesky_factor(self, n, tangent_vec, base_point, expected):
        result = SPDMatrices.differential_cholesky_factor(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))
        self.assertAllClose(
            gs.all(LowerTriangularMatrices(n).belongs(result)), gs.array(True)
        )

    def test_differential_power(self, power, tangent_vec, base_point, expected):
        result = SPDMatrices.differential_power(
            power, gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_inverse_differential_power(self, power, tangent_vec, base_point, expected):
        result = SPDMatrices.inverse_differential_power(
            power, gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_differential_log(self, tangent_vec, base_point, expected):
        result = SPDMatrices.differential_log(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_inverse_differential_log(self, tangent_vec, base_point, expected):
        result = SPDMatrices.inverse_differential_log(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_differential_exp(self, tangent_vec, base_point, expected):
        result = SPDMatrices.differential_exp(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_inverse_differential_exp(self, tangent_vec, base_point, expected):
        result = SPDMatrices.inverse_differential_exp(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_cholesky_factor_belongs(self, n, mat):
        result = SPDMatrices(n).cholesky_factor(gs.array(mat))
        self.assertAllClose(
            gs.all(PositiveLowerTriangularMatrices(n).belongs(result)), True
        )


class TestSPDMetricAffine(geomstats.tests.TestCase, metaclass=Parametrizer):
    class TestDataSPDMetricAffine(TestData):
        def inner_product_data(self):
            smoke_data = [
                dict(
                    n=3,
                    power_affine=0.5,
                    tangent_vec_a=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                    tangent_vec_b=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                    base_point=[[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]],
                    expected=713 / 144,
                )
            ]
            return self.generate_tests(smoke_data)

        def exp_data(self):
            smoke_data = [
                dict(
                    n=2,
                    power_affine=1.0,
                    tangent_vec=[[2.0, 0.0], [0.0, 2.0]],
                    base_point=[[1.0, 0.0], [0.0, 1.0]],
                    expected=[[EXP_2, 0.0], [0.0, EXP_2]],
                )
            ]
            return self.generate_tests(smoke_data)

        def log_data(self):
            smoke_data = [
                dict(
                    n=2,
                    power_affine=1.0,
                    point=[[1.0, 0.0], [0.0, 1.0]],
                    base_point=[[2.0, 0.0], [0.0, 2.0]],
                    expected=[[-2 * LN_2, 0.0], [0.0, -2 * LN_2]],
                )
            ]
            return self.generate_tests(smoke_data)

        def log_exp_composition_data(self):
            power_affine = [1.0, 0.5, -0.5]
            return self._log_exp_composition_data(
                SPDMatrices, power_affine=power_affine
            )

        def geodesic_belongs_data(self):
            power_affine = [1.0]
            return self._geodesic_belongs_data(SPDMatrices, power_affine=power_affine)

        def squared_dist_is_symmetric_data(self):
            power_affine = [1.0, 0.5, -0.5]
            return self._squared_dist_is_symmetric_data(
                SPDMatrices, power_affine=power_affine
            )

        def parallel_transport_exp_norm_data(self):
            random_n = random.sample(range(1, 10), 5)
            random_power_affine = [1.0]
            random_data = [
                dict(n=n, power_affine=power_affine, n_samples=200)
                for (n, power_affine) in zip(random_n, random_power_affine)
            ]
            return self.generate_tests([], random_data)

    testing_data = TestDataSPDMetricAffine()

    def test_inner_product(
        self, n, power_affine, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = SPDMetricAffine(n, power_affine)
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, expected)

    def test_exp(self, n, power_affine, tangent_vec, base_point, expected):
        metric = SPDMetricAffine(n, power_affine)
        self.assertAllClose(
            metric.exp(gs.array(tangent_vec), gs.array(base_point)), gs.array(expected)
        )

    def test_log(self, n, power_affine, point, base_point, expected):
        metric = SPDMetricAffine(n, power_affine)
        self.assertAllClose(
            metric.log(gs.array(point), gs.array(base_point)), gs.array(expected)
        )

    def test_log_exp_composition(self, n, power_affine, point, base_point):
        metric = SPDMetricAffine(n, power_affine)
        log = metric.log(gs.array(point), base_point=gs.array(base_point))
        result = metric.exp(tangent_vec=log, base_point=gs.array(base_point))
        self.assertAllClose(result, point, atol=gs.atol * 1000)

    def test_squared_dist_is_symmetric(self, n, power_affine, point_a, point_b):
        metric = SPDMetricAffine(n, power_affine)
        sd_a_b = metric.squared_dist(gs.array(point_a), gs.array(point_b))
        sd_b_a = metric.squared_dist(gs.array(point_b), gs.array(point_a))
        self.assertAllClose(sd_a_b, sd_b_a, atol=gs.atol * 100)

    def test_parallel_transport_exp_norm(self, n, power_affine, n_samples):
        metric = SPDMetricAffine(n, power_affine)
        space = SPDMatrices(n)
        point = space.random_point(n_samples)
        tan_a = space.random_tangent_vec(n_samples, point)
        tan_b = space.random_tangent_vec(n_samples, point)
        expected = metric.norm(tan_a, point)
        end_point = metric.exp(tan_b, point)

        transported = metric.parallel_transport(tan_a, point, tan_b)
        result = metric.norm(transported, end_point)

        self.assertAllClose(expected, result, gs.atol * 10000)


class TestSPDMetricBuresWasserstein(TestCase, metaclass=Parametrizer):
    class TestDataSPDMetricBuresWasserstein(TestData):
        def inner_product_data(self):
            smoke_data = [
                dict(
                    n=3,
                    tangent_vec_a=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                    tangent_vec_b=[[1.0, 2.0, 4.0], [2.0, 3.0, 8.0], [4.0, 8.0, 5.0]],
                    base_point=[[1.0, 0.0, 0.0], [0.0, 1.5, 0.5], [0.0, 0.5, 1.5]],
                    expected=4.0,
                )
            ]
            return self.generate_tests(smoke_data)

        def exp_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec=[[2.0, 0.0], [0.0, 2.0]],
                    base_point=[[1.0, 0.0], [0.0, 1.0]],
                    expected=[[4.0, 0.0], [0.0, 4.0]],
                )
            ]
            return self.generate_tests(smoke_data)

        def log_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point=[[4.0, 0.0], [0.0, 4.0]],
                    base_point=[[1.0, 0.0], [0.0, 1.0]],
                    expected=[[2.0, 0.0], [0.0, 2.0]],
                )
            ]
            return self.generate_tests(smoke_data)

        def squared_dist_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point_a=[[1.0, 0.0], [0.0, 1.0]],
                    point_b=[[2.0, 0.0], [0.0, 2.0]],
                    expected=2 + 4 - (2 * 2 * SQRT_2),
                )
            ]
            return self.generate_tests(smoke_data)

        def log_exp_composition_data(self):
            return self._log_exp_composition_data(SPDMatrices)

        def geodesic_belongs_data(self):
            return self._geodesic_belongs_data(
                SPDMatrices,
            )

        def squared_dist_is_symmetric_data(self):
            return self._squared_dist_is_symmetric_data(SPDMatrices)

    testing_data = TestDataSPDMetricBuresWasserstein()

    def test_inner_product(self, n, tangent_vec_a, tangent_vec_b, base_point, expected):
        metric = SPDMetricBuresWasserstein(n)
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_exp(self, n, tangent_vec, base_point, expected):
        metric = SPDMetricBuresWasserstein(n)
        result = metric.exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, n, point, base_point, expected):
        metric = SPDMetricBuresWasserstein(n)
        result = metric.log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_squared_dist(self, n, point_a, point_b, expected):
        metric = SPDMetricBuresWasserstein(n)
        result = metric.squared_dist(gs.array(point_a), gs.array(point_b))
        self.assertAllClose(result, gs.array(expected))

    def test_log_exp_composition(self, n, point, base_point):
        metric = SPDMetricBuresWasserstein(n)
        log = metric.log(gs.array(point), base_point=gs.array(base_point))
        result = metric.exp(tangent_vec=log, base_point=gs.array(base_point))
        self.assertAllClose(result, point, atol=gs.atol * 1000)

    def test_squared_dist_is_symmetric(self, n, point_a, point_b):
        metric = SPDMetricBuresWasserstein(n)
        sd_a_b = metric.squared_dist(point_a, point_b)
        sd_b_a = metric.squared_dist(point_b, point_a)
        self.assertAllClose(sd_a_b, sd_b_a, atol=gs.atol * 100)


class TestSPDMetricEuclidean(TestCase, metaclass=Parametrizer):
    class TestDataSPDMetricEuclidean(TestData):
        def inner_product_data(self):
            smoke_data = [
                dict(
                    n=3,
                    power_euclidean=0.5,
                    tangent_vec_a=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                    tangent_vec_b=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                    base_point=[[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]],
                    expected=3472 / 576,
                )
            ]
            return self.generate_tests(smoke_data)

        def exp_domain_data(self):
            smoke_data = [
                dict(
                    n=3,
                    power_euclidean=1.0,
                    tangent_vec=[[-1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.0]],
                    base_point=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                    expected=[-3, 1],
                )
            ]
            return self.generate_tests(smoke_data)

        def exp_data(self):
            smoke_data = [
                dict(
                    n=2,
                    power_euclidean=1.0,
                    tangent_vec=[[2.0, 0.0], [0.0, 2.0]],
                    base_point=[[1.0, 0.0], [0.0, 1.0]],
                    expected=[[3.0, 0.0], [0.0, 3.0]],
                )
            ]
            return self.generate_tests(smoke_data)

        def log_data(self):
            smoke_data = [
                dict(
                    n=2,
                    power_euclidean=1.0,
                    point=[[2.0, 0.0], [0.0, 2.0]],
                    base_point=[[1.0, 0.0], [0.0, 1.0]],
                    expected=[[1.0, 0.0], [0.0, 1.0]],
                )
            ]
            return self.generate_tests(smoke_data)

        def log_exp_composition_data(self):
            power_euclidean = [1.0, 0.5, -0.5]
            return self._log_exp_composition_data(
                SPDMatrices, power_euclidean=power_euclidean
            )

        def geodesic_belongs_data(self):
            power_euclidean = [1.0]
            return self._geodesic_belongs_data(
                SPDMatrices, max_n=3, n_n=2, n_t=5, power_euclidean=power_euclidean
            )

        def squared_dist_is_symmetric_data(self):
            power_euclidean = [1.0]
            return self._squared_dist_is_symmetric_data(
                SPDMatrices, power_euclidean=power_euclidean
            )

    testing_data = TestDataSPDMetricEuclidean()

    def test_inner_product(
        self, n, power_euclidean, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = SPDMetricEuclidean(n, power_euclidean)
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    @geomstats.tests.np_autograd_and_tf_only
    def test_exp_domain(self, n, power_euclidean, tangent_vec, base_point, expected):
        metric = SPDMetricEuclidean(n, power_euclidean)
        result = metric.exp_domain(
            gs.array(tangent_vec), gs.array(base_point), expected
        )
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, n, power_euclidean, point, base_point, expected):
        metric = SPDMetricEuclidean(n)
        result = metric.log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_log_exp_composition(self, n, power_euclidean, point, base_point):
        metric = SPDMetricEuclidean(n, power_euclidean)
        log = metric.log(gs.array(point), base_point=gs.array(base_point))
        result = metric.exp(tangent_vec=log, base_point=gs.array(base_point))
        self.assertAllClose(result, point, atol=gs.atol * 1000)

    def test_squared_dist_is_symmetric(self, n, power_euclidean, point_a, point_b):
        metric = SPDMetricEuclidean(n, power_euclidean)
        sd_a_b = metric.squared_dist(point_a, point_b)
        sd_b_a = metric.squared_dist(point_b, point_a)
        self.assertAllClose(sd_a_b, sd_b_a, atol=gs.atol * 100)


class TestSPDMetricLogEuclidean(geomstats.tests.TestCase, metaclass=Parametrizer):
    class TestDataSPDMetricLogEuclidean(TestData):
        def inner_product_data(self):
            smoke_data = [
                dict(
                    n=3,
                    tangent_vec_a=[[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]],
                    tangent_vec_b=[[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]],
                    base_point=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]],
                    expected=5.0 + (4.0 * ((2 * LN_2) ** 2)),
                )
            ]
            return self.generate_tests(smoke_data)

        def exp_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec=[[2.0, 0.0], [0.0, 2.0]],
                    base_point=[[1.0, 0.0], [0.0, 1.0]],
                    expected=[[EXP_2, 0.0], [0.0, EXP_2]],
                )
            ]
            return self.generate_tests(smoke_data)

        def log_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point=[[2.0, 0.0], [0.0, 2.0]],
                    base_point=[[1.0, 0.0], [0.0, 1.0]],
                    expected=[[LN_2, 0.0], [0.0, LN_2]],
                )
            ]
            return self.generate_tests(smoke_data)

        def log_exp_composition_data(self):
            return self._log_exp_composition_data(SPDMatrices)

        def geodesic_belongs_data(self):
            return self._geodesic_belongs_data(SPDMatrices)

        def squared_dist_is_symmetric_data(self):
            return self._squared_dist_is_symmetric_data(SPDMatrices)

    testing_data = TestDataSPDMetricLogEuclidean()

    def test_inner_product(self, n, tangent_vec_a, tangent_vec_b, base_point, expected):
        metric = SPDMetricLogEuclidean(n)
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_exp(self, n, tangent_vec, base_point, expected):
        metric = SPDMetricLogEuclidean(n)
        result = metric.exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, n, point, base_point, expected):
        metric = SPDMetricLogEuclidean(n)
        result = metric.log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_log_exp_composition(self, n, point, base_point):
        metric = SPDMetricLogEuclidean(n)
        log = metric.log(gs.array(point), base_point=gs.array(base_point))
        result = metric.exp(tangent_vec=log, base_point=gs.array(base_point))
        self.assertAllClose(result, point, atol=gs.atol * 1000)

    def test_squared_dist_is_symmetric(self, n, point_a, point_b):
        metric = SPDMetricLogEuclidean(n)
        sd_a_b = metric.squared_dist(gs.array(point_a), gs.array(point_b))
        sd_b_a = metric.squared_dist(gs.array(point_b), gs.array(point_a))
        self.assertAllClose(sd_a_b, sd_b_a, atol=gs.atol * 100)
