"""Unit tests for Stiefel manifolds."""

import random
import warnings

import pytest
from sklearn import metrics

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric
from tests.conftest import TestCase
from tests.data_generation import LevelSetTestData, RiemannianMetricTestData
from tests.parametrizers import LevelSetParametrizer, RiemannianMetricParametrizer

p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
r_z = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
point1 = gs.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])


class TestStiefel(TestCase, metaclass=LevelSetParametrizer):
    space = Stiefel

    class TestDataStiefel(LevelSetTestData):
        def random_point_belongs_data(self):
            smoke_space_args_list = [(2, 2), (3, 3), (4, 3), (3, 2)]
            smoke_n_points_list = [1, 2, 1, 2]
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            space_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            n_points_list = random.sample(range(1, 10), 5)

            belongs_atol = gs.atol * 1000
            return self._random_point_belongs_data(
                smoke_space_args_list,
                smoke_n_points_list,
                space_args_list,
                n_points_list,
                belongs_atol,
            )

        def to_tangent_is_tangent_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            space_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            tangent_shapes_list = space_args_list
            n_vecs_list = random.sample(range(1, 10), 5)
            is_tangent_atol = gs.atol * 1000
            return self._to_tangent_is_tangent_data(
                Stiefel,
                space_args_list,
                tangent_shapes_list,
                n_vecs_list,
                is_tangent_atol,
            )

        def projection_belongs_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            space_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            shapes_list = space_args_list
            n_samples_list = random.sample(range(1, 10), 5)
            return self._projection_belongs_data(
                space_args_list, shapes_list, n_samples_list
            )

        def to_grassmannian_data(self):

            point1 = gs.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]) / gs.sqrt(2.0)
            batch_points = Matrices.mul(
                GeneralLinear.exp(gs.array([gs.pi * r_z / n for n in [2, 3, 4]])),
                point1,
            )
            smoke_data = [
                dict(point=point1, expected=p_xy),
                dict(point=batch_points, expected=gs.array([p_xy, p_xy, p_xy])),
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataStiefel()

    def test_to_grassmannian(self, point, expected):
        self.assertAllClose(
            self.space.to_grassmannian(gs.array(point)), gs.array(expected)
        )


class TestStiefelCanonicalMetric(TestCase, metaclass=RiemannianMetricParametrizer):
    metric = StiefelCanonicalMetric

    class TestDataStiefelCanonicalMetric(RiemannianMetricTestData):
        def log_two_sheets_error_data(self):
            stiefel = Stiefel(n=3, p=3)
            base_point = stiefel.random_point()
            det_base = gs.linalg.det(base_point)
            point = stiefel.random_point()
            det_point = gs.linalg.det(point)
            if gs.all(det_base * det_point > 0.0):
                point *= -1.0

            random_data = [
                dict(
                    n=3,
                    p=3,
                    point=point,
                    base_point=base_point,
                    expected=pytest.raises(ValueError),
                )
            ]
            return self.generate_tests([], random_data)

        def squared_dist_is_symmetric_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            metric_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            spaces_list = [Stiefel(n, p) for n, p in metric_args_list]
            n_points_a_list = random.sample(range(1, 10), 5)
            n_points_b_list = random.sample(range(1, 10), 5)
            return self._squared_dist_is_symmetric_data(
                metric_args_list, spaces_list, n_points_a_list, n_points_b_list
            )

        def parallel_transport_ivp_is_isometry_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            metric_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            spaces_list = [Stiefel(n, p) for n, p in metric_args_list]
            tangent_shape_list = metric_args_list
            n_tangent_vecs_list = random.sample(range(1, 10), 5)
            n_directions_list = random.sample(range(1, 10), 5)

            return self._parallel_transport_ivp_is_isometry_data(
                metric_args_list,
                spaces_list,
                tangent_shape_list,
                n_tangent_vecs_list,
                n_directions_list,
            )

        def parallel_transport_bvp_is_isometry_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            metric_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            spaces_list = [Stiefel(n, p) for n, p in metric_args_list]
            tangent_shape_list = metric_args_list
            n_base_points_list = random.sample(range(2, 10), 5)
            n_end_points_list = random.sample(range(2, 10), 5)

            return self._parallel_transport_ivp_is_isometry_data(
                metric_args_list,
                spaces_list,
                tangent_shape_list,
                n_base_points_list,
                n_end_points_list,
            )

        def exp_belongs_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            metric_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            spaces_list = [Stiefel(n, p) for n, p in metric_args_list]
            tangent_shapes_list = metric_args_list
            n_tangent_vecs_list = random.sample(range(1, 10), 5)
            return self._exp_belongs_data(
                metric_args_list, spaces_list, n_tangent_vecs_list,  tangent_shapes_list, n_tangent_vecs_list
            )

        def log_is_tangent_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            metric_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            spaces_list = [Stiefel(n, p) for n, p in metric_args_list]
            tangent_shapes_list = metric_args_list
            n_tangent_vecs_list = random.sample(range(1, 10), 5)
            return self._log_is_tangent_data(
                metric_args_list, spaces_list, n_tangent_vecs_list,  tangent_shapes_list, n_tangent_vecs_list
            )

        def geodesic_ivp_belongs_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            metric_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            spaces_list = [Stiefel(n, p) for n, p in metric_args_list]
            n_points_a_list = random.sample(range(1, 10), 5)
            n_points_b_list = random.sample(range(1, 10), 5)
            return self._geodesic_ivp_belongs_data()

        def geodesic_bvp_belongs_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            metric_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            spaces_list = [Stiefel(n, p) for n, p in metric_args_list]
            n_points_a_list = random.sample(range(1, 10), 5)
            n_points_b_list = random.sample(range(1, 10), 5)
            return self._geodesic_bvp_belongs_data()

        def log_exp_composition_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            metric_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            spaces_list = [Stiefel(n, p) for n, p in metric_args_list]
            n_points_a_list = random.sample(range(1, 10), 5)
            n_points_b_list = random.sample(range(1, 10), 5)
            return self._log_exp_composition_data()

        def exp_log_composition_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            metric_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            spaces_list = [Stiefel(n, p) for n, p in metric_args_list]
            n_points_a_list = random.sample(range(1, 10), 5)
            n_points_b_list = random.sample(range(1, 10), 5)
            return self._exp_log_composition_data()

        def exp_ladder_parallel_transport_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            metric_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            spaces_list = [Stiefel(n, p) for n, p in metric_args_list]
            n_points_a_list = random.sample(range(1, 10), 5)
            n_points_b_list = random.sample(range(1, 10), 5)
            return self._exp_ladder_parallel_transport_data()

        def exp_geodesic_ivp_data(self):
            n_list = random.sample(range(2, 10), 5)
            p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
            metric_args_list = [(n, p) for n, p in zip(n_list, p_list)]
            spaces_list = [Stiefel(n, p) for n, p in metric_args_list]
            n_points_a_list = random.sample(range(1, 10), 5)
            n_points_b_list = random.sample(range(1, 10), 5)
            return self._exp_geodesic_ivp_data()

    testing_data = TestDataStiefelCanonicalMetric()

    def test_log_two_sheets_error(self, n, p, point, base_point, expected):
        metric = self.space(n, p)
        with expected:
            metric.log(point, base_point)
