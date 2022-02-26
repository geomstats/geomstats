"""Unit tests for landmarks space."""

import random

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.landmarks import L2Metric, Landmarks
from tests.conftest import TestCase
from tests.data_generation import ManifoldTestData, RiemannianMetricTestData
from tests.parametrizers import ManifoldParametrizer, RiemannianMetricParametrizer


class TestLandmarks(TestCase, metaclass=ManifoldParametrizer):
    space = Landmarks
    skip_test_random_point_belongs = True

    class TestDataLandmarks(ManifoldTestData):
        dim_list = random.sample(range(2, 4), 2)
        n_landmarks_list = random.sample(range(1, 5), 2)
        space_args_list = [
            (Hypersphere(dim), n_landmarks)
            for dim, n_landmarks in zip(dim_list, n_landmarks_list)
        ] + [
            (Euclidean(dim + 1), n_landmarks)
            for dim, n_landmarks in zip(dim_list, n_landmarks_list)
        ]
        shape_list = [
            (n_landmark, dim + 1) for dim, n_landmark in zip(dim_list, n_landmarks_list)
        ] * 2
        n_points_list = random.sample(range(1, 7), 4)
        n_samples_list = random.sample(range(1, 7), 4)
        n_vecs_list = random.sample(range(2, 5), 2)

        def random_point_belongs_data(self):
            smoke_space_args_list = [(Hypersphere(2), 2), (Euclidean(2 + 1), 2)]
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
                Landmarks,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

    testing_data = TestDataLandmarks()


class TestL2Metric(TestCase, metaclass=RiemannianMetricParametrizer):
    metric = connection = L2Metric
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_belongs = True
    skip_test_exp_log_composition = True
    skip_test_exp_shape = True
    skip_test_log_shape = True

    class TestDataL2Metric(RiemannianMetricTestData):

        dim_list = random.sample(range(1, 3), 2)
        n_landmarks_list = random.sample(range(2, 5), 2)
        metric_args_list = [
            (Hypersphere(dim), n_landmarks)
            for dim, n_landmarks in zip(dim_list, n_landmarks_list)
        ] + [
            (Euclidean(dim + 1), n_landmarks)
            for dim, n_landmarks in zip(dim_list, n_landmarks_list)
        ]
        space_list = [Landmarks(*metric_arg) for metric_arg in metric_args_list]
        shape_list = [
            (n_landmark, dim + 1) for dim, n_landmark in zip(dim_list, n_landmarks_list)
        ] * 2
        n_points_list = random.sample(range(2, 7), 4)
        n_samples_list = random.sample(range(2, 7), 4)
        n_points_a_list = random.sample(range(2, 7), 4)
        n_points_b_list = [1]
        batch_size_list = random.sample(range(2, 7), 4)
        alpha_list = [1] * 4
        n_rungs_list = [1] * 4
        scheme_list = ["pole"] * 4

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
                belongs_atol=gs.atol * 10000,
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
                belongs_atol=gs.atol * 100,
            )

        def log_exp_composition_data(self):
            return self._log_exp_composition_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
            )

        def exp_log_composition_data(self):
            return self._exp_log_composition_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                rtol=gs.rtol * 10000,
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
                rtol=gs.rtol * 10000,
                atol=gs.atol * 10000,
            )

        def parallel_transport_ivp_is_isometry_data(self):
            return self._parallel_transport_ivp_is_isometry_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 100,
            )

        def parallel_transport_bvp_is_isometry_data(self):
            return self._parallel_transport_bvp_is_isometry_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                is_tangent_atol=gs.atol * 100,
                atol=gs.atol * 100,
            )

    testing_data = TestDataL2Metric()
