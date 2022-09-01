import random

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.landmarks import L2LandmarksMetric, Landmarks
from tests.data_generation import _ManifoldTestData, _RiemannianMetricTestData


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

    Space = Landmarks

    def random_point_belongs_test_data(self, belongs_atol=gs.atol):
        space_args_list = [(Hypersphere(2), 2), (Euclidean(2 + 1), 2)]
        n_points_list = [1, 2]
        random_data = [
            dict(space_args=space_args, n_points=n_points, belongs_atol=belongs_atol)
            for space_args, n_points in zip(space_args_list, n_points_list)
        ]
        return self.generate_tests([], random_data)

    def dimension_is_dim_multiplied_by_n_copies_test_data(self):
        smoke_data = [
            dict(space_args=space_args) for space_args in self.space_args_list
        ]
        return self.generate_tests(smoke_data)


class TestDataL2LandmarksMetric(_RiemannianMetricTestData):

    dim_list = random.sample(range(2, 4), 2)
    n_landmarks_list = random.sample(range(2, 5), 2)
    space_args_list = [
        (Hypersphere(dim), n_landmarks)
        for dim, n_landmarks in zip(dim_list, n_landmarks_list)
    ] + [
        (Euclidean(dim + 1), n_landmarks)
        for dim, n_landmarks in zip(dim_list, n_landmarks_list)
    ]
    metric_args_list = [
        (space.metric, n_landmarks) for space, n_landmarks in space_args_list
    ]
    space_list = [Landmarks(*space_arg) for space_arg in space_args_list]
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

    initial_point = gs.array([0.0, 0.0, 1.0])
    initial_tangent_vec_a = gs.array([1.0, 0.0, 0.0])
    initial_tangent_vec_b = gs.array([0.0, 1.0, 0.0])
    initial_tangent_vec_c = gs.array([-1.0, 0.0, 0.0])

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

    Metric = L2LandmarksMetric

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=30.0)

    def l2_metric_inner_product_vectorization_test_data(self):
        smoke_data = [
            dict(
                l2_metric_s2=self.l2_metric_s2,
                times=self.times,
                n_landmark_sets=self.n_landmark_sets,
                landmarks_a=self.landmark_set_a,
                landmarks_b=self.landmark_set_b,
                landmarks_c=self.landmark_set_c,
            )
        ]
        return self.generate_tests(smoke_data)

    def l2_metric_exp_vectorization_test_data(self):
        smoke_data = [
            dict(
                l2_metric_s2=self.l2_metric_s2,
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
                l2_metric_s2=self.l2_metric_s2,
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
                l2_metric_s2=self.l2_metric_s2,
                times=self.times,
                n_sampling_points=self.n_sampling_points,
                landmarks_a=self.landmark_set_a,
                landmarks_b=self.landmark_set_b,
            )
        ]
        return self.generate_tests(smoke_data)

    def innerproduct_is_sum_of_innerproducts_test_data(self):
        smoke_data = [
            dict(
                metric_args=(Hypersphere(dim=2).metric, self.n_sampling_points),
                tangent_vec_a=self.landmark_set_a,
                tangent_vec_b=self.landmark_set_b,
                base_point=self.landmark_set_c,
                rtol=gs.rtol,
                atol=gs.atol,
            )
        ]
        return self.generate_tests(smoke_data)
