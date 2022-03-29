"""Unit tests for landmarks space."""

import random

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.landmarks import L2Metric, Landmarks
from tests.conftest import Parametrizer
from tests.data_generation import _ManifoldTestData, _RiemannianMetricTestData
from tests.geometry_test_cases import ManifoldTestCase, RiemannianMetricTestCase


class TestLandmarks(ManifoldTestCase, metaclass=Parametrizer):
    space = Landmarks
    skip_test_random_point_belongs = True
    skip_test_random_tangent_vec_is_tangent = True

    class TestDataLandmarks(_ManifoldTestData):
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
        n_points_list = random.sample(range(1, 5), 4)
        n_vecs_list = random.sample(range(2, 5), 2)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(Hypersphere(2), 2), (Euclidean(2 + 1), 2)]
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
                Landmarks,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                Landmarks, self.space_args_list, self.n_vecs_list
            )

    testing_data = TestDataLandmarks()


class TestL2Metric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = L2Metric
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_belongs = True
    skip_test_exp_shape = True
    skip_test_log_shape = True

    class TestDataL2Metric(_RiemannianMetricTestData):

        dim_list = random.sample(range(2, 4), 2)
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
        n_points_list = random.sample(range(2, 5), 2)
        n_tangent_vecs_list = random.sample(range(2, 5), 2)
        n_points_a_list = random.sample(range(2, 5), 2)
        n_points_b_list = [1]
        alpha_list = [1] * 4
        n_rungs_list = [1] * 4
        scheme_list = ["pole"] * 4

        s2 = Hypersphere(dim=2)
        r3 = s2.embedding_space

        initial_point = [0.0, 0.0, 1.0]
        initial_tangent_vec_a = [1.0, 0.0, 0.0]
        initial_tangent_vec_b = [0.0, 1.0, 0.0]
        initial_tangent_vec_c = [-1.0, 0.0, 0.0]

        landmarks_a = s2.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec_a
        )
        landmarks_b = s2.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec_b
        )
        landmarks_c = s2.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec_c
        )

        n_sampling_points = 10
        sampling_times = gs.linspace(0.0, 1.0, n_sampling_points)
        landmark_set_a = landmarks_a(sampling_times)
        landmark_set_b = landmarks_b(sampling_times)
        landmark_set_c = landmarks_c(sampling_times)

        n_landmark_sets = 5
        times = gs.linspace(0.0, 1.0, n_landmark_sets)
        space_landmarks_in_sphere_2d = Landmarks(
            ambient_manifold=s2, k_landmarks=n_sampling_points
        )
        l2_metric_s2 = space_landmarks_in_sphere_2d.metric

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
                belongs_atol=gs.atol * 10000,
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
                belongs_atol=gs.atol * 100,
            )

        def log_then_exp_test_data(self):
            return self._log_then_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_tangent_vecs_list,
                rtol=gs.rtol * 1000,
                atol=gs.atol * 1000,
            )

        def exp_then_log_test_data(self):
            return self._exp_then_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                amplitude=30,
                rtol=gs.rtol * 10000,
                atol=gs.atol * 100000,
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
                atol=gs.atol * 100,
            )

        def parallel_transport_bvp_is_isometry_test_data(self):
            return self._parallel_transport_bvp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 100,
                atol=gs.atol * 100,
            )

        def l2_metric_inner_product_vectorization_test_data(self):
            smoke_data = [
                dict(
                    l2_metric=self.l2_metric_s2,
                    times=self.times,
                    landmark_sets=self.n_landmark_sets,
                    landmarks_a=self.landmark_set_a,
                    landmarks_b=self.landmark_set_b,
                    landmarks_c=self.landmark_set_c,
                )
            ]
            return self.generate_tests(smoke_data)

        def l2_metric_exp_vectorization_test_data(self):
            smoke_data = [
                dict(
                    l2_metric=self.l2_metric_s2,
                    times=self.times,
                    landmarks_a=self.landmark_set_a,
                    landmarks_b=self.landmark_set_b,
                    landmarks_c=self.landmark_set_c,
                )
            ]
            return self.generate_tests(smoke_data)

        def l2_metric_log_vectorization_test_data(self):
            smoke_data = [
                dict(
                    l2_metric=self.l2_metric_s2,
                    times=self.times,
                    landmarks_a=self.landmark_set_a,
                    landmarks_b=self.landmark_set_b,
                    landmarks_c=self.landmark_set_c,
                )
            ]
            return self.generate_tests(smoke_data)

        def l2_metric_geodesic_test_data(self):
            smoke_data = [
                dict(
                    l2_metric=self.l2_metric_s2,
                    times=self.times,
                    n_sampling_points=self.n_sampling_points,
                    landmarks_a=self.landmark_set_a,
                    landmarks_b=self.landmark_set_b,
                )
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataL2Metric()

    @geomstats.tests.np_autograd_and_tf_only
    def test_l2_metric_inner_product_vectorization(
        self,
        l2_metric_s2,
        times,
        n_landmark_sets,
        landmarks_a,
        landmarks_b,
        landmarks_c,
    ):
        """Test the vectorization inner_product."""
        landmarks_ab = l2_metric_s2.geodesic(landmarks_a, landmarks_b)
        landmarks_bc = l2_metric_s2.geodesic(landmarks_b, landmarks_c)
        landmarks_ab = landmarks_ab(times)
        landmarks_bc = landmarks_bc(times)

        tangent_vecs = l2_metric_s2.log(point=landmarks_bc, base_point=landmarks_ab)

        result = l2_metric_s2.inner_product(tangent_vecs, tangent_vecs, landmarks_ab)

        self.assertAllClose(gs.shape(result), (n_landmark_sets,))

    @geomstats.tests.np_autograd_and_tf_only
    def test_l2_metric_exp_vectorization(
        self, l2_metric_s2, times, landmarks_a, landmarks_b, landmarks_c
    ):
        """Test the vectorization of exp."""
        landmarks_ab = l2_metric_s2.geodesic(landmarks_a, landmarks_b)
        landmarks_bc = l2_metric_s2.geodesic(landmarks_b, landmarks_c)
        landmarks_ab = landmarks_ab(times)
        landmarks_bc = landmarks_bc(times)

        tangent_vecs = l2_metric_s2.log(point=landmarks_bc, base_point=landmarks_ab)

        result = l2_metric_s2.exp(tangent_vec=tangent_vecs, base_point=landmarks_ab)
        self.assertAllClose(gs.shape(result), gs.shape(landmarks_ab))

    @geomstats.tests.np_autograd_and_tf_only
    def test_l2_metric_log_vectorization(
        self, l2_metric_s2, times, landmarks_a, landmarks_b, landmarks_c
    ):
        """Test the vectorization of log."""
        landmarks_ab = l2_metric_s2.geodesic(landmarks_a, landmarks_b)
        landmarks_bc = l2_metric_s2.geodesic(landmarks_b, landmarks_c)
        landmarks_ab = landmarks_ab(times)
        landmarks_bc = landmarks_bc(times)

        tangent_vecs = l2_metric_s2.log(point=landmarks_bc, base_point=landmarks_ab)

        result = tangent_vecs
        self.assertAllClose(gs.shape(result), gs.shape(landmarks_ab))

    @geomstats.tests.np_autograd_and_tf_only
    def test_l2_metric_geodesic(
        self, l2_metric_s2, times, n_sampling_points, landmarks_a, landmarks_b
    ):
        """Test the geodesic method of L2Metric."""
        landmarks_ab = l2_metric_s2.geodesic(landmarks_a, landmarks_b)
        landmarks_ab = landmarks_ab(times)

        result = landmarks_ab
        expected = []
        for k in range(n_sampling_points):
            geod = l2_metric_s2.ambient_metric.geodesic(
                initial_point=landmarks_a[k, :], end_point=landmarks_b[k, :]
            )
            expected.append(geod(times))
        expected = gs.stack(expected, axis=1)

        self.assertAllClose(result, expected)
