import random

import geomstats.backend as gs
from geomstats.geometry.poincare_ball import PoincareBall, PoincareBallMetric
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData


class PoincareBallTestData(_OpenSetTestData):
    smoke_space_args_list = [(2,), (3,), (4,), (5,)]
    smoke_n_points_list = [1, 2, 1, 2]
    n_list = random.sample(range(2, 10), 5)
    space_args_list = [(n,) for n in n_list]
    n_points_list = random.sample(range(1, 10), 5)
    shape_list = [(n,) for n in n_list]
    n_vecs_list = random.sample(range(1, 10), 5)

    Space = PoincareBall

    def belongs_test_data(self):
        smoke_data = [
            dict(dim=2, point=[0.3, 0.5], expected=True),
            dict(dim=2, point=[1.2, 0.5], expected=False),
        ]
        return self.generate_tests(smoke_data)

    def projection_norm_lessthan_1_test_data(self):
        smoke_data = [dict(dim=2, point=[1.2, 0.5])]
        return self.generate_tests(smoke_data)


class TestDataPoincareBallMetric(_RiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)
    metric_args_list = [(n,) for n in n_list]
    shape_list = [(n,) for n in n_list]
    space_list = [PoincareBall(n) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = PoincareBallMetric

    def mobius_out_of_the_ball_test_data(self):
        smoke_data = [dict(dim=2, x=[0.7, 0.9], y=[0.2, 0.2])]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point=[0.3, 0.5],
                base_point=[0.3, 0.3],
                expected=[-0.01733576, 0.21958634],
            )
        ]
        return self.generate_tests(smoke_data)

    def dist_pairwise_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.5]],
                expected=[
                    [0.0, 0.65821943, 1.34682524],
                    [0.65821943, 0.0, 0.71497076],
                    [1.34682524, 0.71497076, 0.0],
                ],
            )
        ]
        return self.generate_tests(smoke_data)

    def dist_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point_a=[0.5, 0.5],
                point_b=[0.5, -0.5],
                expected=2.887270927429199,
            )
        ]
        return self.generate_tests(smoke_data)

    def coordinate_test_data(self):
        smoke_data = [dict(dim=2, point_a=[-0.3, 0.7], point_b=[0.2, 0.5])]
        return self.generate_tests(smoke_data)

    def mobius_vectorization_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point_a=gs.array([0.5, 0.5]),
                point_b=gs.array([[0.5, -0.3], [0.3, 0.4]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_vectorization_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point_a=gs.array([0.5, 0.5]),
                point_b=gs.array([[0.5, -0.5], [0.4, 0.4]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_vectorization_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point_a=gs.array([0.5, 0.5]),
                point_b=gs.array([[0.0, 0.0], [0.5, -0.5], [0.4, 0.4]]),
            )
        ]
        return self.generate_tests(smoke_data)
