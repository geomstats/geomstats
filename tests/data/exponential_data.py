import random

import geomstats.backend as gs
from geomstats.information_geometry.exponential import (
    ExponentialDistributions,
    ExponentialMetric,
)
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData


class ExponentialTestData(_OpenSetTestData):
    Space = ExponentialDistributions
    n_list = random.sample((2, 5), 1)
    n_samples_list = random.sample(range(10), 3)
    space_args_list = []
    metric_args_list = []
    shape_list = [(1,)]
    n_points_list = random.sample(range(5), 2)
    n_vecs_list = random.sample(range(2, 5), 1)

    def belongs_test_data(self):
        smoke_data = [
            dict(point=gs.array([5.0]), expected=True),
            dict(point=gs.array([-2.0]), expected=False),
            dict(point=gs.array([[1.0], [-1.0]]), expected=gs.array([True, False])),
            dict(point=gs.array([[0.1], [10]]), expected=gs.array([True, True])),
            dict(point=gs.array([[-2.1], [-1.0]]), expected=gs.array([False, False])),
        ]
        return self.generate_tests(smoke_data)

    def random_point_test_data(self):
        random_data = [
            dict(point=self.Space().random_point(3), expected=(3, 1)),
            dict(point=self.Space().random_point(2), expected=(2, 1)),
            dict(point=self.Space().random_point(1), expected=(1,)),
        ]
        return self.generate_tests([], random_data)

    def sample_shape_test_data(self):
        smoke_data = [
            dict(point=gs.array([[0.2], [0.3]]), n_samples=2, expected=(2, 2)),
            dict(
                point=gs.array([[0.1], [0.2], [0.3]]),
                n_samples=1,
                expected=(3, 1),
            ),
        ]
        return self.generate_tests(smoke_data)

    def sample_belongs_test_data(self):
        random_data = [
            dict(
                point=self.Space().random_point(3),
                n_samples=4,
                expected=gs.ones((3, 4)),
            ),
            dict(
                point=self.Space().random_point(1),
                n_samples=2,
                expected=gs.ones(2),
            ),
            dict(
                point=self.Space().random_point(2),
                n_samples=3,
                expected=gs.ones((2, 3)),
            ),
        ]
        return self.generate_tests([], random_data)

    def point_to_pdf_test_data(self):
        random_data = [
            dict(
                point=self.Space().random_point(1),
                n_samples=4,
            ),
            dict(
                point=self.Space().random_point(1),
                n_samples=1,
            ),
            dict(
                point=self.Space().random_point(4),
                n_samples=1,
            ),
            dict(
                point=self.Space().random_point(4),
                n_samples=4,
            ),
        ]
        return self.generate_tests([], random_data)


class ExponentialMetricTestData(_RiemannianMetricTestData):
    Space = ExponentialDistributions
    Metric = ExponentialMetric

    n_list = random.sample(range(2, 5), 2)
    n_samples_list = random.sample(range(1, 10), 3)
    connection_args_list = metric_args_list = [() for n in n_list]
    space_list = [ExponentialDistributions() for n in n_list]
    space_args_list = [() for n in n_list]
    shape_list = [(1,) for n in n_list]
    n_points_a_list = n_points_b_list = n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = n_vecs_list = random.sample(range(2, 5), 2)

    tolerances = {
        "dist_point_to_itself_is_zero": {"atol": 1e-5},
        "dist_is_symmetric": {"atol": 5e-1},
        "dist_is_norm_of_log": {"atol": 5e-1},
        "exp_subspace": {"atol": 1e-4},
        "triangle_inequality_of_dist": {"atol": 1e-10},
    }

    def squared_dist_test_data(self):
        smoke_data = [
            dict(
                point_a=gs.array([1, 0.5, 10]),
                point_b=gs.array([2, 3.5, 70]),
                expected=gs.array([0.48045301, 3.78656631, 3.78656631]),
            ),
            dict(
                point_a=gs.array(0.1),
                point_b=gs.array(0.99),
                expected=gs.array(5.255715612697455),
            ),
            dict(
                point_a=gs.array(0.1),
                point_b=gs.array([0.99, 0.2]),
                expected=gs.array([5.255715612697455, 0.48045301]),
            ),
            dict(
                point_a=gs.array([0.99, 0.2]),
                point_b=gs.array(0.1),
                expected=gs.array([5.255715612697455, 0.48045301]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def metric_matrix_test_data(self):
        smoke_data = [
            dict(
                point=gs.array([0.5]),
                expected=gs.array([[4.0]]),
            ),
            dict(
                point=gs.array([[0.5], [0.2]]),
                expected=gs.array([[[4.0]], [[25.0]]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def geodesic_symmetry_test_data(self):
        random_data = []
        for space_args in self.space_args_list:
            random_data.append(dict(space_args=space_args))
        return self.generate_tests([], random_data)
