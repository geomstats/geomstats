import random

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.data.hypersphere_data import HypersphereMetricTestData
from tests.data_generation import TestData

RTOL = 1e-4
ATOL = 1e-5


class CircleAsSO2Metric(PullbackDiffeoMetric):
    def __init__(self, dim=1):
        # dim is let only to match real Hypersphere signature
        if not dim == 1:
            raise ValueError(
                "This dummy class using SO(2) metric for S1 has "
                "a meaning only when dim=1"
            )
        super(CircleAsSO2Metric, self).__init__(dim=1, shape=(2,))

    def create_embedding_metric(self):
        return SpecialOrthogonal(n=2, point_type="matrix").bi_invariant_metric

    def diffeomorphism(self, base_point):
        second_column = gs.stack([-base_point[..., 1], base_point[..., 0]], axis=-1)
        return gs.stack([base_point, second_column], axis=-1)

    def inverse_diffeomorphism(self, image_point):
        return image_point[..., 0]


class CircleAsSO2(Hypersphere):
    def __init__(self, dim=1):
        if not dim == 1:
            # dim is let only to match real Hypersphere signature
            raise ValueError(
                "This dummy class using SO(2) metric for S1 has "
                "a meaning only when dim=1"
            )
        super(CircleAsSO2, self).__init__(1, "extrinsic")
        self._metric = CircleAsSO2Metric(1)


class CircleAsSO2MetricTestData(HypersphereMetricTestData):
    dim_list = [1] * 4
    metric_args_list = [() for n in dim_list]
    shape_list = [(dim + 1,) for dim in dim_list]
    space_list = [CircleAsSO2(n) for n in dim_list]
    n_points_list = random.sample(range(1, 5), 4)
    n_tangent_vecs_list = random.sample(range(1, 5), 4)
    n_points_a_list = [3] * 4
    n_points_b_list = [1]
    alpha_list = [1] * 4
    n_rungs_list = [1] * 4
    scheme_list = ["pole"] * 4

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                dim=1,
                tangent_vec_a=[1.0, 0.0],
                tangent_vec_b=[2.0, 0.0],
                base_point=[0.0, 1.0],
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
        smoke_data = [dict(dim=1, point_a=point_a, point_b=point_b, expected=gs.pi / 2)]
        return self.generate_tests(smoke_data)

    def diameter_test_data(self):
        point_a = gs.array([[0.0, 1.0]])
        point_b = gs.array([[1.0, 0.0]])
        point_c = gs.array([[1.0, 0.0]])
        smoke_data = [
            dict(
                dim=1, points=gs.vstack((point_a, point_b, point_c)), expected=gs.pi / 2
            )
        ]
        return self.generate_tests(smoke_data)

    def christoffels_shape_test_data(self):
        point = gs.array([[gs.pi / 2], [gs.pi / 6]])
        smoke_data = [dict(dim=1, point=point, expected=[2, 1, 1, 1])]
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
                    dim=dim,
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
                dim=1,
                point=[
                    gs.array([1.0, 0.0]),
                    gs.array([0.0, 1.0]),
                ],
                expected=gs.array([[0.0, gs.pi / 2], [gs.pi / 2, 0.0]]),
                rtol=1e-3,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_after_log_test_data(self):
        # edge case: two very close points, base_point_2 and point_2,
        # form an angle < epsilon
        base_point = gs.array([1.0, 2.0])
        base_point = base_point / gs.linalg.norm(base_point)
        point = base_point + 1e-4 * gs.array([-1.0, -2.0])
        point = point / gs.linalg.norm(point)
        smoke_data = [
            dict(
                space_args=(1,),
                point=point,
                base_point=base_point,
                rtol=gs.rtol,
                atol=gs.atol,
            )
        ]
        return self._exp_after_log_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            smoke_data,
            atol=1e-3,
        )

    def log_after_exp_test_data(self):
        base_point = gs.array([1.0, 0.0])
        tangent_vec = gs.array([0.0, gs.pi / 6])

        smoke_data = [
            dict(
                space_args=(1,),
                tangent_vec=tangent_vec,
                base_point=base_point,
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
            )
        ]
        return self._log_after_exp_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            smoke_data,
            rtol=gs.rtol * 100,
            atol=gs.atol * 100,
        )

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
                dim=1,
                vector=gs.array([0.1, 0.8]),
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)


class CircleAsSO2PullbackDiffeoMetricTestData(TestData):
    def diffeomorphism_is_reciprocal_test_data(self):
        smoke_data = [
            dict(
                metric_args=[],
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
                metric_args=[],
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
