"""Unit tests for Minkowski space."""

import math
import random

import geomstats.backend as gs
from geomstats.geometry.minkowski import Minkowski, MinkowskiMetric
from tests.conftest import TestCase
from tests.data_generation import RiemannianMetricTestData, VectorSpaceTestData
from tests.parametrizers import RiemannianMetricParametrizer, VectorSpaceParametrizer


class TestMinkowski(TestCase, metaclass=VectorSpaceParametrizer):
    space = Minkowski
    skip_test_basis_belongs = True
    skip_test_basis_cardinality = True

    class TestDataMinkowski(VectorSpaceTestData):
        n_list = random.sample(range(2, 8), 2)
        space_args_list = [(n,) for n in n_list]
        shape_list = space_args_list
        n_samples_list = random.sample(range(2, 8), 2)
        n_points_list = random.sample(range(2, 8), 2)
        n_vecs_list = random.sample(range(2, 8), 2)

        def belongs_data(self):
            smoke_data = [dict(dim=2, point=[-1.0, 3.0], expected=True)]
            return self.generate_tests(smoke_data)

        def basis_belongs_data(self):
            return self._basis_belongs_data(self.space_args_list)

        def basis_cardinality_data(self):
            return self._basis_cardinality_data(self.space_args_list)

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
                Minkowski,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

    testing_data = TestDataMinkowski()

    def test_belongs(self, dim, point, expected):
        self.assertAllClose(
            self.space(dim).belongs(gs.array(point)), gs.array(expected)
        )


class TestMinkowskiMetric(TestCase, metaclass=RiemannianMetricParametrizer):
    connection = metric = MinkowskiMetric
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True

    class TestDataMinkowskiMetric(RiemannianMetricTestData):
        n_list = random.sample(range(2, 7), 5)
        metric_args_list = [(n,) for n in n_list]
        shape_list = metric_args_list
        space_list = [Minkowski(n) for n in n_list]
        n_points_list = random.sample(range(1, 7), 5)
        n_samples_list = random.sample(range(1, 7), 5)
        n_points_a_list = random.sample(range(1, 7), 5)
        n_points_b_list = [1]
        batch_size_list = random.sample(range(2, 7), 5)
        alpha_list = [1] * 5
        n_rungs_list = [1] * 5
        scheme_list = ["pole"] * 5

        def metric_matrix_data(self):
            smoke_data = [dict(dim=2, expected=[[-1.0, 0.0], [0.0, 1.0]])]
            return self.generate_tests(smoke_data)

        def inner_product_data(self):
            smoke_data = [
                dict(dim=2, point_a=[0.0, 1.0], point_b=[2.0, 10.0], expected=10.0),
                dict(
                    dim=2,
                    point_a=[[-1.0, 0.0], [1.0, 0.0], [2.0, math.sqrt(3)]],
                    point_b=[
                        [2.0, -math.sqrt(3)],
                        [4.0, math.sqrt(15)],
                        [-4.0, math.sqrt(15)],
                    ],
                    expected=[2.0, -4.0, 14.70820393],
                ),
            ]
            return self.generate_tests(smoke_data)

        def squared_norm_data(self):
            smoke_data = [dict(dim=2, vector=[-2.0, 4.0], expected=12.0)]
            return self.generate_tests(smoke_data)

        def squared_dist_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    point_a=[2.0, -math.sqrt(3)],
                    point_b=[4.0, math.sqrt(15)],
                    expected=27.416407,
                )
            ]
            return self.generate_tests(smoke_data)

        def exp_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    tangent_vec=[2.0, math.sqrt(3)],
                    base_point=[1.0, 0.0],
                    expected=[3.0, math.sqrt(3)],
                )
            ]
            return self.generate_tests(smoke_data)

        def log_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    point=[2.0, math.sqrt(3)],
                    base_point=[-1.0, 0.0],
                    expected=[3.0, math.sqrt(3)],
                )
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
                rtol=gs.rtol * 1000,
                atol=gs.atol * 1000,
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

    testing_data = TestDataMinkowskiMetric()

    def test_metric_matrix(self, dim, expected):
        metric = self.metric(dim)
        self.assertAllClose(metric.metric_matrix(), gs.array(expected))

    def test_inner_product(self, dim, point_a, point_b, expected):
        metric = self.metric(dim)
        self.assertAllClose(
            metric.inner_product(gs.array(point_a), gs.array(point_b)),
            gs.array(expected),
        )

    def test_squared_norm(self, dim, point, expected):
        metric = self.metric(dim)
        self.assertAllClose(metric.squared_norm(gs.array(point)), gs.array(expected))

    def test_exp(self, dim, tangent_vec, base_point, expected):
        result = self.metric(dim).exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, dim, point, base_point, expected):
        result = self.metric(dim).log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_squared_dist(self, dim, point_a, point_b, expected):
        result = self.metric(dim).squared_dist(gs.array(point_a), gs.array(point_b))
        self.assertAllClose(result, gs.array(expected))
