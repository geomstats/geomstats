"""Unit tests for the Grassmannian."""
import random

import geomstats.backend as gs
from geomstats.geometry.grassmannian import Grassmannian, GrassmannianCanonicalMetric
from geomstats.geometry.matrices import Matrices
from tests.conftest import Parametrizer
from tests.data_generation import _LevelSetTestData, _RiemannianMetricTestData
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase

p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
p_yz = gs.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
p_xz = gs.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

r_y = gs.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
r_z = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
pi_2 = gs.pi / 2
pi_4 = gs.pi / 4


class TestGrassmannian(LevelSetTestCase, metaclass=Parametrizer):
    space = Grassmannian
    skip_test_extrinsic_then_intrinsic = True
    skip_test_intrinsic_then_extrinsic = True

    class GrassmannianTestData(_LevelSetTestData):
        n_list = random.sample(range(3, 6), 2)
        k_list = [random.sample(range(2, n), 1)[0] for n in n_list]
        space_args_list = list(zip(n_list, k_list))
        shape_list = [(n, n) for n in n_list]
        n_vecs_list = random.sample(range(1, 5), 2)
        n_points_list = random.sample(range(1, 5), 2)

        def belongs_test_data(self):
            smoke_data = [
                dict(n=3, k=2, point=p_xy, expected=True),
                dict(n=3, k=2, point=gs.array([p_yz, p_xz]), expected=[True, True]),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(3, 2), (4, 2)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
                belongs_atol=1e-3,
            )

        def to_tangent_is_tangent_test_data(self):

            is_tangent_atol = gs.atol * 1000
            return self._to_tangent_is_tangent_test_data(
                Grassmannian,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
                is_tangent_atol,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

    testing_data = GrassmannianTestData()

    def test_belongs(self, n, k, point, expected):
        self.assertAllClose(self.space(n, k).belongs(point), gs.array(expected))


class TestGrassmannianCanonicalMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = GrassmannianCanonicalMetric
    skip_test_exp_then_log = True
    skip_test_exp_geodesic_ivp = True

    class GrassmannianCanonicalMetricTestData(_RiemannianMetricTestData):
        n_list = random.sample(range(3, 5), 2)
        k_list = [random.sample(range(2, n), 1)[0] for n in n_list]
        metric_args_list = list(zip(n_list, k_list))
        shape_list = [(n, n) for n in n_list]
        space_list = [Grassmannian(n, p) for n, p in metric_args_list]
        n_points_list = random.sample(range(1, 5), 2)
        n_points_a_list = random.sample(range(1, 5), 2)
        n_points_b_list = [1]
        n_tangent_vecs_list = random.sample(range(1, 5), 2)
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

        def exp_test_data(self):
            smoke_data = [
                dict(
                    n=3,
                    k=2,
                    tangent_vec=Matrices.bracket(pi_2 * r_y, gs.array([p_xy, p_yz])),
                    base_point=gs.array([p_xy, p_yz]),
                    expected=gs.array([p_yz, p_xy]),
                ),
                dict(
                    n=3,
                    k=2,
                    tangent_vec=Matrices.bracket(
                        pi_2 * gs.array([r_y, r_z]), gs.array([p_xy, p_yz])
                    ),
                    base_point=gs.array([p_xy, p_yz]),
                    expected=gs.array([p_yz, p_xz]),
                ),
            ]
            return self.generate_tests(smoke_data)

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(
                self.metric_args_list,
                self.space_list,
            )

        def squared_dist_is_symmetric_test_data(self):
            return self._squared_dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
                atol=gs.atol * 1000,
            )

        def exp_belongs_test_data(self):
            return self._exp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_is_tangent_test_data(self):
            return self._log_is_tangent_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                is_tangent_atol=gs.atol * 1000,
            )

        def geodesic_ivp_belongs_test_data(self):
            return self._geodesic_ivp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 10000,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 10000,
            )

        def log_then_exp_test_data(self):
            return self._log_then_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def exp_then_log_test_data(self):
            return self._exp_then_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def exp_ladder_parallel_transport_test_data(self):
            return self._exp_ladder_parallel_transport_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_rungs_list,
                self.alpha_list,
                self.scheme_list,
            )

        def exp_geodesic_ivp_test_data(self):
            return self._exp_geodesic_ivp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_points_list,
                rtol=gs.rtol * 10000,
                atol=gs.atol * 10000,
            )

        def parallel_transport_ivp_is_isometry_test_data(self):
            return self._parallel_transport_ivp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_test_data(self):
            return self._parallel_transport_bvp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

    testing_data = GrassmannianCanonicalMetricTestData()

    def test_exp(self, n, k, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.metric(n, k).exp(gs.array(tangent_vec), gs.array(base_point)),
            gs.array(expected),
        )
