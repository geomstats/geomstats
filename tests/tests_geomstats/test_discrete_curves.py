"""Unit tests for parameterized manifolds."""

import random

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import (
    ClosedDiscreteCurves,
    ClosedSRVMetric,
    DiscreteCurves,
    L2CurvesMetric,
    SRVMetric,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from tests.conftest import Parametrizer
from tests.data_generation import _ManifoldTestData, _RiemannianMetricTestData
from tests.geometry_test_cases import ManifoldTestCase, RiemannianMetricTestCase


class TestDiscreteCurves(ManifoldTestCase, metaclass=Parametrizer):
    space = DiscreteCurves
    skip_test_projection_belongs = True
    skip_test_random_tangent_vec_is_tangent = True

    class DiscreteCurvesTestData(_ManifoldTestData):
        s2 = Hypersphere(dim=2)
        r2 = Euclidean(dim=2)
        r3 = Euclidean(dim=3)
        space_args_list = [(r2,), (r3,)]
        shape_list = [(10, 2), (10, 3)]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(self.s2,), (self.r2,)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list, self.shape_list, self.n_points_list
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                DiscreteCurves,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                DiscreteCurves, self.space_args_list, self.n_vecs_list
            )

    testing_data = DiscreteCurvesTestData()


class TestL2CurvesMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = L2CurvesMetric
    skip_test_exp_belongs = True
    skip_test_exp_shape = True
    skip_test_log_shape = True
    skip_test_exp_geodesic_ivp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True

    class L2CurvesMetricTestData(_RiemannianMetricTestData):
        s2 = Hypersphere(dim=2)
        r2 = Euclidean(dim=2)
        r3 = Euclidean(dim=3)
        ambient_manifolds_list = [r2, r3]
        metric_args_list = [
            (ambient_manifolds,) for ambient_manifolds in ambient_manifolds_list
        ]
        shape_list = [(10, 2), (10, 3)]
        space_list = [
            DiscreteCurves(ambient_manifolds)
            for ambient_manifolds in ambient_manifolds_list
        ]
        n_points_list = random.sample(range(2, 5), 2)
        n_tangent_vecs_list = random.sample(range(2, 5), 2)
        n_points_a_list = random.sample(range(2, 5), 2)
        n_points_b_list = [1]
        batch_size_list = random.sample(range(2, 5), 2)
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(self.metric_args_list, self.space_list)

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
                belongs_atol=gs.atol * 1000,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def exp_after_log_test_data(self):
            return self._exp_after_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def log_after_exp_test_data(self):
            return self._log_after_exp_test_data(
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
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
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

        def dist_is_symmetric_test_data(self):
            return self._dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_positive_test_data(self):
            return self._dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def squared_dist_is_positive_test_data(self):
            return self._squared_dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_norm_of_log_test_data(self):
            return self._dist_is_norm_of_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_point_to_itself_is_zero_test_data(self):
            return self._dist_point_to_itself_is_zero_test_data(
                self.metric_args_list, self.space_list, self.n_points_list
            )

        def inner_product_is_symmetric_test_data(self):
            return self._inner_product_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
            )

    testing_data = L2CurvesMetricTestData()


class TestSRVMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = SRVMetric
    skip_test_exp_shape = True
    skip_test_log_shape = True
    skip_test_exp_geodesic_ivp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_geodesic_ivp_belongs = True

    class SRVMetricTestData(_RiemannianMetricTestData):
        s2 = Hypersphere(dim=2)
        r2 = Euclidean(dim=2)
        r3 = Euclidean(dim=3)
        ambient_manifolds_list = [r2, r3]
        metric_args_list = [
            (ambient_manifolds,) for ambient_manifolds in ambient_manifolds_list
        ]
        shape_list = [(10, 2), (10, 3)]
        space_list = [
            DiscreteCurves(ambient_manifolds)
            for ambient_manifolds in ambient_manifolds_list
        ]
        n_points_list = random.sample(range(2, 5), 2)
        n_tangent_vecs_list = random.sample(range(2, 5), 2)
        n_points_a_list = [1, 2]
        n_points_b_list = [1, 2]
        batch_size_list = random.sample(range(2, 5), 2)
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(self.metric_args_list, self.space_list)

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
                belongs_atol=gs.atol * 1000,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def exp_after_log_test_data(self):
            return self._exp_after_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def log_after_exp_test_data(self):
            return self._log_after_exp_test_data(
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
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
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

        def dist_is_symmetric_test_data(self):
            return self._dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_positive_test_data(self):
            return self._dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def squared_dist_is_positive_test_data(self):
            return self._squared_dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_norm_of_log_test_data(self):
            return self._dist_is_norm_of_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_point_to_itself_is_zero_test_data(self):
            return self._dist_point_to_itself_is_zero_test_data(
                self.metric_args_list, self.space_list, self.n_points_list
            )

        def inner_product_is_symmetric_test_data(self):
            return self._inner_product_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
            )

    testing_data = SRVMetricTestData()


class TestClosedDiscreteCurves(ManifoldTestCase, metaclass=Parametrizer):
    space = ClosedDiscreteCurves
    skip_test_projection_belongs = True
    skip_test_random_point_belongs = True
    skip_test_random_tangent_vec_is_tangent = True
    skip_test_to_tangent_is_tangent = True

    class DiscreteCurvesTestData(_ManifoldTestData):
        s2 = Hypersphere(dim=2)
        r2 = Euclidean(dim=2)
        r3 = Euclidean(dim=3)
        space_args_list = [(r2,), (r3,)]
        shape_list = [(10, 2), (10, 3)]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(self.s2,), (self.r2,)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list, self.shape_list, self.n_points_list
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                DiscreteCurves,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                DiscreteCurves, self.space_args_list, self.n_vecs_list
            )

    testing_data = DiscreteCurvesTestData()


class ClosedSRVMetric(RiemannianMetricTestCase, metaclass=Parametrizer):

    metric = connection = ClosedSRVMetric
    skip_test_exp_shape = True
    skip_test_log_shape = True
    skip_test_exp_geodesic_ivp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_exp_belongs = True
    skip_test_squared_dist_is_symmetric = True
    skip_test_log_is_tangent = True
    skip_test_dist_is_positive = True
    skip_test_dist_is_symmetric = True
    skip_test_squared_dist_is_symmetric = True
    skip_test_squared_dist_is_positive = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_inner_product_is_symmetric = True
    skip_test_dist_is_norm_of_log = True
    skip_test_exp_after_log = True
    skip_test_log_after_exp = True
    skip_test_exp_ladder_parallel_transport = True

    class ClosedSRVMetricTestData(_RiemannianMetricTestData):
        s2 = Hypersphere(dim=2)
        r2 = Euclidean(dim=2)
        r3 = Euclidean(dim=3)
        ambient_manifolds_list = [r2, r3]
        metric_args_list = [
            (ambient_manifolds,) for ambient_manifolds in ambient_manifolds_list
        ]
        shape_list = [(10, 2), (10, 3)]
        space_list = [
            ClosedDiscreteCurves(ambient_manifolds)
            for ambient_manifolds in ambient_manifolds_list
        ]
        n_points_list = random.sample(range(2, 5), 2)
        n_tangent_vecs_list = random.sample(range(2, 5), 2)
        n_points_a_list = [1, 2]
        n_points_b_list = [1, 2]
        batch_size_list = random.sample(range(2, 5), 2)
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(self.metric_args_list, self.space_list)

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
                belongs_atol=gs.atol * 1000,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def exp_after_log_test_data(self):
            return self._exp_after_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def log_after_exp_test_data(self):
            return self._log_after_exp_test_data(
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
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
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

        def dist_is_symmetric_test_data(self):
            return self._dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_positive_test_data(self):
            return self._dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def squared_dist_is_positive_test_data(self):
            return self._squared_dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_norm_of_log_test_data(self):
            return self._dist_is_norm_of_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_point_to_itself_is_zero_test_data(self):
            return self._dist_point_to_itself_is_zero_test_data(
                self.metric_args_list, self.space_list, self.n_points_list
            )

        def inner_product_is_symmetric_test_data(self):
            return self._inner_product_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
            )

    testing_data = ClosedSRVMetricTestData()
