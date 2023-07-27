import random

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.invariant_metric import BiInvariantMetric
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.data.hypersphere_data import HypersphereMetricTestData
from tests.data_generation import _RiemannianMetricTestData

RTOL = 1e-4
ATOL = 1e-5


class CircleAsSO2Metric(PullbackDiffeoMetric):
    def __init__(self, space):
        if not space.dim == 1:
            raise ValueError(
                "This dummy class using SO(2) metric for S1 has "
                "a meaning only when dim=1"
            )
        super().__init__(space=space)

    def _define_embedding_space(self):
        space = SpecialOrthogonal(n=2, point_type="matrix", equip=False)
        space.equip_with_metric(BiInvariantMetric)
        return space

    def diffeomorphism(self, base_point):
        second_column = gs.stack([-base_point[..., 1], base_point[..., 0]], axis=-1)
        return gs.stack([base_point, second_column], axis=-1)

    def inverse_diffeomorphism(self, image_point):
        return image_point[..., 0]


class CircleAsSO2MetricTestData(HypersphereMetricTestData):
    metric_args_list = [{}]
    shape_list = [(2,)]
    space_list = [Hypersphere(dim=1, equip=False)]

    n_points_list = random.sample(range(1, 5), 4)
    n_tangent_vecs_list = random.sample(range(1, 5), 4)
    n_points_a_list = [3] * 4
    n_points_b_list = [1]
    alpha_list = [1] * 4
    n_rungs_list = [1] * 4
    scheme_list = ["pole"] * 4

    Metric = CircleAsSO2Metric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                space=self.space_list[0],
                tangent_vec_a=gs.array([1.0, 0.0]),
                tangent_vec_b=gs.array([2.0, 0.0]),
                base_point=gs.array([0.0, 1.0]),
                expected=2.0,
            )
        ]
        return self.generate_tests(smoke_data)

    def dist_test_data(self):
        # smoke data is currently testing points at orthogonal
        point_a = gs.array([10.0, -2.0])
        point_a = point_a / gs.linalg.norm(point_a)
        point_b = gs.array([2.0, 10])
        point_b = point_b / gs.linalg.norm(point_b)
        smoke_data = [
            dict(
                space=self.space_list[0],
                point_a=point_a,
                point_b=point_b,
                expected=gs.pi / 2,
            )
        ]
        return self.generate_tests(smoke_data)

    def diameter_test_data(self):
        point_a = gs.array([[0.0, 1.0]])
        point_b = gs.array([[1.0, 0.0]])
        point_c = gs.array([[1.0, 0.0]])
        smoke_data = [
            dict(
                space=self.space_list[0],
                points=gs.vstack((point_a, point_b, point_c)),
                expected=gs.pi / 2,
            )
        ]
        return self.generate_tests(smoke_data)

    def christoffels_shape_test_data(self):
        point = gs.array([[gs.pi / 2], [gs.pi / 6]])
        smoke_data = [
            dict(space=self.space_list[0], point=point, expected=[2, 1, 1, 1])
        ]
        return self.generate_tests(smoke_data)

    def sectional_curvature_test_data(self):
        dim_list = [1]
        n_samples_list = random.sample(range(1, 4), 2)
        random_data = []
        for dim, n_samples in zip(dim_list, n_samples_list):
            sphere = Hypersphere(dim)
            base_point = sphere.random_uniform()
            tangent_vec_a = sphere.to_tangent(
                gs.random.rand(n_samples, sphere.dim + 1), base_point
            )
            tangent_vec_b = sphere.to_tangent(
                gs.random.rand(n_samples, sphere.dim + 1), base_point
            )
            expected = gs.ones(n_samples)  # try shape here
            random_data.append(
                dict(
                    space=self.space_list[0],
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                ),
            )
        return self.generate_tests(random_data)

    def dist_pairwise_test_data(self):
        smoke_data = [
            dict(
                space=self.space_list[0],
                point=gs.array(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                ),
                expected=gs.array([[0.0, gs.pi / 2], [gs.pi / 2, 0.0]]),
                rtol=1e-3,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_after_log_smoke_test_data(self):
        # edge case: two very close points, base_point_2 and point_2,
        # form an angle < epsilon
        base_point = gs.array([1.0, 2.0])
        base_point = base_point / gs.linalg.norm(base_point)
        point = base_point + 1e-4 * gs.array([-1.0, -2.0])
        point = point / gs.linalg.norm(point)
        smoke_data = [
            dict(
                space=self.space_list[0],
                connection_args={},
                point=point,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_smoke_test_data(self):
        base_point = gs.array([1.0, 0.0])
        tangent_vec = gs.array([0.0, gs.pi / 6])

        smoke_data = [
            dict(
                space=self.space_list[0],
                connection_args={},
                tangent_vec=tangent_vec,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_and_dist_and_projection_to_tangent_space_test_data(self):
        unnorm_base_point = gs.array(
            [
                16.0,
                -2.0,
            ]
        )
        base_point = unnorm_base_point / gs.linalg.norm(unnorm_base_point)
        smoke_data = [
            dict(
                space=self.space_list[0],
                vector=gs.array([0.1, 0.8]),
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_shape_test_data(self):
        index = 0
        metric_args = self.metric_args_list[index]
        space = self.space_list[index]

        n = 2
        base_points = space.random_point(n)
        tangent_vecs = space.to_tangent(space.random_point(n), base_points)

        tangent_vecs_0 = space.to_tangent(space.random_point(2), base_points[0])

        data = [
            [space, metric_args, tangent_vecs[0], base_points[0], base_points[0].shape],
            [space, metric_args, tangent_vecs, base_points, base_points.shape],
            [space, metric_args, tangent_vecs_0, base_points[0], tangent_vecs_0.shape],
        ]

        return self.generate_tests([], data)


class CircleAsSO2PullbackDiffeoMetricTestData(_RiemannianMetricTestData):
    Metric = CircleAsSO2Metric
    space_list = [Hypersphere(dim=1, equip=False)]
    metric_args_list = []
    n_points_a_list = []
    n_points_b_list = []
    n_points_list = []
    shape_list = []
    n_tangent_vecs_list = []
    connection_args_list = []
    n_rungs_list = []
    alpha_list = []
    scheme_list = []

    def diffeomorphism_is_reciprocal_test_data(self):
        smoke_data = [
            dict(
                space=self.space_list[0],
                metric_args={},
                point=gs.array(
                    [
                        [1.0, 0.0],
                        [0.7648421873, -0.6442176872],
                        [0.0, -1.0],
                    ]
                ),
                rtol=RTOL,
                atol=ATOL,
            ),
        ]
        return self.generate_tests(smoke_data)

    def tangent_diffeomorphism_is_reciprocal_test_data(self):
        smoke_data = [
            dict(
                space=self.space_list[0],
                metric_args={},
                point=gs.array(
                    [
                        [1.0, 0.0],
                        [0.7648421873, 0.6442176872],
                        [0.0, -1.0],
                    ]
                ),
                tangent_vector=gs.array(
                    [
                        [0.0, 2.0],
                        [0.3221088436, 0.3824210936],
                        [0.5, 0.0],
                    ]
                ),
                rtol=RTOL,
                atol=ATOL,
            ),
        ]
        return self.generate_tests(smoke_data)

    def matrix_innerproduct_and_embedded_innerproduct_coincide_test_data(self):
        smoke_data = []
        return self.generate_tests(smoke_data)
