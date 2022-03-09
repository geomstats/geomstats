"""Unit tests for Positive lower triangular matrices"""


import math
import random

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.positive_lower_triangular_matrices import (
    CholeskyMetric,
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.conftest import TestCase
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData
from tests.parametrizers import OpenSetParametrizer, RiemannianMetricParametrizer

EULER = gs.exp(1.0)
SQRT_2 = math.sqrt(2)


class TestPositiveLowerTriangularMatrices(TestCase, metaclass=OpenSetParametrizer):
    """Test of Cholesky methods."""

    space = PositiveLowerTriangularMatrices

    class PositiveLowerTriangularMatricesTestData(_OpenSetTestData):
        n_list = random.sample(range(2, 5), 2)
        space_args_list = [(n,) for n in n_list]
        shape_list = [(n, n) for n in n_list]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_data(self):
            smoke_data = [
                dict(n=2, mat=[[1.0, 0.0], [-1.0, 3.0]], expected=True),
                dict(n=2, mat=[[1.0, -1.0], [-1.0, 3.0]], expected=False),
                dict(n=2, mat=[[-1.0, 0.0], [-1.0, 3.0]], expected=False),
                dict(n=3, mat=[[1.0, 0], [0, 1.0]], expected=False),
                dict(
                    n=2,
                    mat=[
                        [[1.0, 0], [0, 1.0]],
                        [[1.0, 2.0], [2.0, 1.0]],
                        [[-1.0, 0.0], [1.0, 1.0]],
                        [[0.0, 0.0], [1.0, 1.0]],
                    ],
                    expected=[True, False, False, False],
                ),
                dict(
                    n=3,
                    mat=[
                        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[-1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    ],
                    expected=[False, False, True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def random_point_and_belongs_data(self):
            smoke_data = [
                dict(n=1, n_samples=1),
                dict(n=2, n_samples=2),
                dict(n=10, n_samples=100),
                dict(n=100, n_samples=10),
            ]
            return self.generate_tests(smoke_data)

        def gram_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point=[[1.0, 0.0], [2.0, 1.0]],
                    expected=[[1.0, 2.0], [2.0, 5.0]],
                ),
                dict(
                    n=2,
                    point=[[[2.0, 1.0], [0.0, 1.0]], [[-6.0, 0.0], [5.0, 3.0]]],
                    expected=[[[5.0, 1.0], [1.0, 1.0]], [[36.0, -30.0], [-30.0, 34.0]]],
                ),
            ]
            return self.generate_tests(smoke_data)

        def differential_gram_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec=[[-1.0, 0.0], [2.0, -1.0]],
                    base_point=[[1.0, 0.0], [2.0, 1.0]],
                    expected=[[-2.0, 0.0], [0.0, 6.0]],
                ),
                dict(
                    n=2,
                    tangent_vec=[[[-1.0, 2.0], [2.0, -1.0]], [[0.0, 4.0], [4.0, -1.0]]],
                    base_point=[[[3.0, 0.0], [-1.0, 2.0]], [[4.0, 0.0], [-1.0, 4.0]]],
                    expected=[
                        [[-6.0, 11.0], [11.0, -8.0]],
                        [[0.0, 32.0], [32.0, -16.0]],
                    ],
                ),
            ]
            return self.generate_tests(smoke_data)

        def inverse_differential_gram_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec=[[1.0, 2.0], [2.0, 5.0]],
                    base_point=[[1.0, 0.0], [2.0, 2.0]],
                    expected=[[0.5, 0.0], [1.0, 0.25]],
                ),
                dict(
                    n=2,
                    tangent_vec=[[[-4.0, 1.0], [1.0, -4.0]], [[0.0, 4.0], [4.0, -8.0]]],
                    base_point=[[[2.0, 0.0], [-1.0, 2.0]], [[4.0, 0.0], [-1.0, 2.0]]],
                    expected=[[[-1.0, 0.0], [0.0, -1.0]], [[0.0, 0.0], [1.0, -1.5]]],
                ),
            ]
            return self.generate_tests(smoke_data)

        def differential_gram_belongs_data(self):
            n_list = [1, 2, 2, 3, 10]
            n_samples_list = [1, 1, 2, 10, 5]
            space = PositiveLowerTriangularMatrices
            random_data = [
                dict(
                    n=n,
                    tangent_vec=space(n).ambient_space.random_point(n_samples),
                    base_point=space(n).random_point(n_samples),
                )
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def inverse_differential_gram_belongs_data(self):
            n_list = [1, 2, 2, 3, 10]
            n_samples_list = [1, 1, 2, 10, 5]
            space = PositiveLowerTriangularMatrices
            random_data = [
                dict(
                    n=n,
                    tangent_vec=space(n).ambient_space.random_point(n_samples),
                    base_point=space(n).random_point(n_samples),
                )
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def random_point_belongs_data(self):
            smoke_space_args_list = [(2,), (3,)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_data(self):
            return self._projection_belongs_data(
                self.space_args_list, self.shape_list, self.n_samples_list
            )

        def to_tangent_is_tangent_data(self):
            return self._to_tangent_is_tangent_data(
                PositiveLowerTriangularMatrices,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def to_tangent_is_tangent_in_ambient_space_data(self):
            return self._to_tangent_is_tangent_in_ambient_space_data(
                PositiveLowerTriangularMatrices, self.space_args_list, self.shape_list
            )

    testing_data = PositiveLowerTriangularMatricesTestData()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.space(n).belongs(gs.array(mat)), gs.array(expected))

    def test_gram(self, n, point, expected):
        self.assertAllClose(self.space(n).gram(gs.array(point)), gs.array(expected))

    def test_differential_gram(self, n, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.space(n).differential_gram(
                gs.array(tangent_vec), gs.array(base_point)
            ),
            gs.array(expected),
        )

    def test_inverse_differential_gram(self, n, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.space(n).inverse_differential_gram(
                gs.array(tangent_vec), gs.array(base_point)
            ),
            gs.array(expected),
        )

    @geomstats.tests.np_and_autograd_only
    def test_differential_gram_belongs(self, n, tangent_vec, base_point):
        result = self.space(n).differential_gram(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(gs.all(SymmetricMatrices(n).belongs(result)), True)

    def test_inverse_differential_gram_belongs(self, n, tangent_vec, base_point):
        result = self.space(n).inverse_differential_gram(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(gs.all(self.space(n).ambient_space.belongs(result)), True)


class TestCholeskyMetric(TestCase, metaclass=RiemannianMetricParametrizer):
    metric = connection = CholeskyMetric
    space = PositiveLowerTriangularMatrices

    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True

    class TestDataCholeskyMetric(RiemannianMetricTestData):
        n_list = random.sample(range(2, 7), 5)
        metric_args_list = [(n,) for n in n_list]
        shape_list = [(n, n) for n in n_list]
        space_list = [PositiveLowerTriangularMatrices(n) for n in n_list]
        n_points_list = random.sample(range(1, 7), 5)
        n_samples_list = random.sample(range(1, 7), 5)
        n_points_a_list = random.sample(range(1, 7), 5)
        n_points_b_list = [1]
        batch_size_list = random.sample(range(2, 7), 5)
        alpha_list = [1] * 5
        n_rungs_list = [1] * 5
        scheme_list = ["pole"] * 5

        def diag_inner_product_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec_a=[[1.0, 0.0], [-2.0, -1.0]],
                    tangent_vec_b=[[2.0, 0.0], [-3.0, -1.0]],
                    base_point=[[SQRT_2, 0.0], [-3.0, 1.0]],
                    expected=2.0,
                )
            ]
            return self.generate_tests(smoke_data)

        def strictly_lower_inner_product_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec_a=[[1.0, 0.0], [-2.0, -1.0]],
                    tangent_vec_b=[[2.0, 0.0], [-3.0, -1.0]],
                    expected=6.0,
                )
            ]
            return self.generate_tests(smoke_data)

        def inner_product_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec_a=[[1.0, 0.0], [-2.0, -1.0]],
                    tangent_vec_b=[[2.0, 0.0], [-3.0, -1.0]],
                    base_point=[[SQRT_2, 0.0], [-3.0, 1.0]],
                    expected=8.0,
                ),
                dict(
                    n=2,
                    tangent_vec_a=[
                        [[3.0, 0.0], [4.0, 2.0]],
                        [[-1.0, 0.0], [2.0, -4.0]],
                    ],
                    tangent_vec_b=[[[4.0, 0.0], [3.0, 3.0]], [[3.0, 0.0], [-6.0, 2.0]]],
                    base_point=[[[3, 0.0], [-2.0, 6.0]], [[1, 0.0], [-1.0, 1.0]]],
                    expected=[13.5, -23.0],
                ),
            ]
            return self.generate_tests(smoke_data)

        def exp_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec=[[-1.0, 0.0], [2.0, 3.0]],
                    base_point=[[1.0, 0.0], [2.0, 2.0]],
                    expected=[[1 / EULER, 0.0], [4.0, 2 * gs.exp(1.5)]],
                ),
                dict(
                    n=2,
                    tangent_vec=[[[0.0, 0.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]],
                    base_point=[[[1.0, 0.0], [2.0, 2.0]], [[1.0, 0.0], [0.0, 2.0]]],
                    expected=[
                        [[1.0, 0.0], [4.0, 2.0]],
                        [[gs.exp(1.0), 0.0], [0.0, 2.0]],
                    ],
                ),
            ]
            return self.generate_tests(smoke_data)

        def log_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point=[[EULER, 0.0], [2.0, EULER**3]],
                    base_point=[[EULER**3, 0.0], [4.0, EULER**4]],
                    expected=[[-2.0 * EULER**3, 0.0], [-2.0, -1 * EULER**4]],
                ),
                dict(
                    n=2,
                    point=[
                        [[gs.exp(-2.0), 0.0], [0.0, gs.exp(2.0)]],
                        [[gs.exp(-3.0), 0.0], [2.0, gs.exp(3.0)]],
                    ],
                    base_point=[[[1.0, 0.0], [-1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                    expected=[[[-2.0, 0.0], [1.0, 2.0]], [[-3.0, 0.0], [2.0, 3.0]]],
                ),
            ]
            return self.generate_tests(smoke_data)

        def squared_dist_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point_a=[[EULER, 0.0], [2.0, EULER**3]],
                    point_b=[[EULER**3, 0.0], [4.0, EULER**4]],
                    expected=9,
                ),
                dict(
                    n=2,
                    point_a=[
                        [[EULER, 0.0], [2.0, EULER**3]],
                        [[EULER, 0.0], [4.0, EULER**3]],
                    ],
                    point_b=[
                        [[EULER**3, 0.0], [4.0, EULER**4]],
                        [[EULER**3, 0.0], [7.0, EULER**4]],
                    ],
                    expected=[9, 14],
                ),
            ]
            return self.generate_tests(smoke_data)

        def exp_shape_data(self):
            return self._exp_shape_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.batch_size_list,
            )

        def log_shape_data(self):
            return self._log_shape_data(
                self.metric_args_list,
                self.space_list,
                self.batch_size_list,
            )

        def squared_dist_is_symmetric_data(self):
            return self._squared_dist_is_symmetric_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
                atol=gs.atol * 1000,
            )

        def exp_belongs_data(self):
            return self._exp_belongs_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_is_tangent_data(self):
            return self._log_is_tangent_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
                is_tangent_atol=gs.atol * 1000,
            )

        def geodesic_ivp_belongs_data(self):
            return self._geodesic_ivp_belongs_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def geodesic_bvp_belongs_data(self):
            return self._geodesic_bvp_belongs_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_exp_composition_data(self):
            return self._log_exp_composition_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def exp_log_composition_data(self):
            return self._exp_log_composition_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def exp_ladder_parallel_transport_data(self):
            return self._exp_ladder_parallel_transport_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                self.n_rungs_list,
                self.alpha_list,
                self.scheme_list,
            )

        def exp_geodesic_ivp_data(self):
            return self._exp_geodesic_ivp_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                self.n_points_list,
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
            )

        def parallel_transport_ivp_is_isometry_data(self):
            return self._parallel_transport_ivp_is_isometry_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_data(self):
            return self._parallel_transport_bvp_is_isometry_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

    testing_data = TestDataCholeskyMetric()

    def test_diag_inner_product(
        self, n, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        result = self.metric(n).diag_inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_strictly_lower_inner_product(
        self, n, tangent_vec_a, tangent_vec_b, expected
    ):
        result = self.metric(n).strictly_lower_inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_inner_product(self, n, tangent_vec_a, tangent_vec_b, base_point, expected):
        result = self.metric(n).inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_exp(self, n, tangent_vec, base_point, expected):
        result = self.metric(n).exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, n, point, base_point, expected):
        result = self.metric(n).log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_squared_dist(self, n, point_a, point_b, expected):
        result = self.metric(n).squared_dist(gs.array(point_a), gs.array(point_b))
        self.assertAllClose(result, gs.array(expected))
