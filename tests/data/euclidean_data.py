import math
import random

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from tests.data_generation import _RiemannianMetricTestData, _VectorSpaceTestData

SQRT_2 = math.sqrt(2)
SQRT_5 = math.sqrt(5)


class EuclideanTestData(_VectorSpaceTestData):

    Space = Euclidean

    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    shape_list = [(n,) for n in n_list]
    n_points_list = random.sample(range(2, 5), 2)
    batch_shape_list = [
        tuple(random.choices(range(2, 10), k=i)) for i in random.sample(range(1, 5), 3)
    ]
    n_vecs_list = random.sample(range(2, 5), 2)

    def belongs_test_data(self):
        smoke_data = [
            dict(dim=2, vec=[0.0, 1.0], expected=True),
            dict(dim=2, vec=[1.0, 0.0, 1.0], expected=False),
        ]
        return self.generate_tests(smoke_data)


class EuclideanMetricTestData(_RiemannianMetricTestData):
    n_list = random.sample(range(2, 7), 5)
    connection_args_list = metric_args_list = [(n,) for n in n_list]
    shape_list = metric_args_list
    space_list = [Euclidean(n) for n in n_list]
    n_points_list = random.sample(range(1, 7), 5)
    n_tangent_vecs_list = n_vecs_list = random.sample(range(1, 7), 5)
    n_points_a_list = random.sample(range(1, 7), 5)
    n_points_b_list = [1]
    alpha_list = [1] * 5
    n_rungs_list = [1] * 5
    scheme_list = ["pole"] * 5

    Metric = EuclideanMetric

    def exp_test_data(self):

        one_tangent_vec = gs.array([0.0, 1.0])
        one_base_point = gs.array([2.0, 10.0])
        n_tangent_vecs = gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]])
        n_base_points = gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]])
        smoke_data = [
            dict(
                dim=2,
                tangent_vec=[0.0, 1.0],
                base_point=[2.0, 10.0],
                expected=[2.0, 11.0],
            ),
            dict(
                dim=2,
                tangent_vec=one_tangent_vec,
                base_point=one_base_point,
                expected=one_tangent_vec + one_base_point,
            ),
            dict(
                dim=2,
                tangent_vec=one_tangent_vec,
                base_point=n_base_points,
                expected=one_tangent_vec + n_base_points,
            ),
            dict(
                dim=2,
                tangent_vec=n_tangent_vecs,
                base_point=one_base_point,
                expected=n_tangent_vecs + one_base_point,
            ),
            dict(
                dim=2,
                tangent_vec=n_tangent_vecs,
                base_point=n_base_points,
                expected=n_tangent_vecs + n_base_points,
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        one_p = gs.array([0.0, 1.0])
        one_bp = gs.array([2.0, 10.0])
        n_ps = gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]])
        n_bps = gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]])
        smoke_data = [
            dict(dim=2, point=[2.0, 10.0], base_point=[0.0, 1.0], expected=[2.0, 9.0]),
            dict(dim=2, point=one_p, base_point=one_bp, expected=one_p - one_bp),
            dict(dim=2, point=one_p, base_point=n_bps, expected=one_p - n_bps),
            dict(dim=2, point=n_ps, base_point=one_bp, expected=n_ps - one_bp),
            dict(dim=2, point=n_ps, base_point=n_bps, expected=n_ps - n_bps),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_test_data(self):
        n_tangent_vecs_1 = [[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]]
        n_tangent_vecs_2 = [[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]]
        tangent_vec_3 = [0.0, 1.0]
        tangent_vec_4 = [2.0, 10.0]
        smoke_data = [
            dict(
                dim=2,
                tangent_vec_a=n_tangent_vecs_1,
                tangent_vec_b=tangent_vec_4,
                expected=[14.0, -44.0, 0.0],
            ),
            dict(
                dim=2,
                tangent_vec_a=tangent_vec_3,
                tangent_vec_b=n_tangent_vecs_2,
                expected=[10.0, -1.0, 6.0],
            ),
            dict(
                dim=2,
                tangent_vec_a=n_tangent_vecs_1,
                tangent_vec_b=n_tangent_vecs_2,
                expected=[14.0, -12.0, 21.0],
            ),
            dict(
                dim=2,
                tangent_vec_a=[0.0, 1.0],
                tangent_vec_b=[2.0, 10.0],
                expected=10.0,
            ),
        ]
        return self.generate_tests(smoke_data)

    def squared_norm_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                vec=[0.0, 1.0],
                expected=1.0,
            ),
            dict(
                dim=2,
                vec=[[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]],
                expected=[5.0, 20.0, 26.0],
            ),
        ]
        return self.generate_tests(smoke_data)

    def norm_test_data(self):
        smoke_data = [
            dict(dim=2, vec=[4.0, 3.0], expected=5.0),
            dict(dim=4, vec=[4.0, 3.0, 4.0, 3.0], expected=5.0 * SQRT_2),
            dict(
                dim=3,
                vec=[[4.0, 3.0, 10.0], [3.0, 10.0, 4.0]],
                expected=[5 * SQRT_5, 5 * SQRT_5],
            ),
        ]
        return self.generate_tests(smoke_data)

    def metric_matrix_test_data(self):
        smoke_data = [
            dict(dim=1, expected=gs.eye(1)),
            dict(dim=2, expected=gs.eye(2)),
            dict(dim=3, expected=gs.eye(3)),
        ]
        return self.generate_tests(smoke_data)

    def squared_dist_test_data(self):
        one_point_a = gs.array([0.0, 1.0])
        one_point_b = gs.array([2.0, 10.0])
        n_points_a = gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]])
        n_points_b = gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]])
        smoke_data = []
        smoke_data.append(
            dict(
                dim=2,
                point_a=one_point_a,
                point_b=n_points_b,
                expected=[85.0, 68.0, 34.0],
            )
        )
        smoke_data.append(
            dict(dim=2, point_a=one_point_a, point_b=one_point_b, expected=85.0)
        )
        smoke_data.append(
            dict(
                dim=2,
                point_a=n_points_a,
                point_b=one_point_b,
                expected=[81.0, 212.0, 130.0],
            )
        )
        smoke_data.append(
            dict(
                dim=2,
                point_a=n_points_a,
                point_b=n_points_b,
                expected=[81.0, 109.0, 29.0],
            )
        )

        return self.generate_tests(smoke_data)

    def dist_test_data(self):
        one_point_a = gs.array([0.0, 1.0])
        one_point_b = gs.array([2.0, 10.0])
        n_points_a = gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]])
        n_points_b = gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]])
        smoke_data = []
        smoke_data.append(
            dict(
                dim=2,
                point_a=one_point_a,
                point_b=n_points_b,
                expected=gs.sqrt(gs.array([85.0, 68.0, 34.0])),
            )
        )
        smoke_data.append(
            dict(
                dim=2,
                point_a=one_point_a,
                point_b=one_point_b,
                expected=gs.sqrt(gs.array(85.0)),
            )
        )
        smoke_data.append(
            dict(
                dim=2,
                point_a=n_points_a,
                point_b=one_point_b,
                expected=gs.sqrt(gs.array([81.0, 212.0, 130.0])),
            )
        )
        smoke_data.append(
            dict(
                dim=2,
                point_a=n_points_a,
                point_b=n_points_b,
                expected=gs.sqrt(gs.array([81.0, 109.0, 29.0])),
            )
        )

        return self.generate_tests(smoke_data)
