import random

import geomstats.backend as gs
from geomstats.information_geometry.poisson import PoissonDistributions, PoissonMetric
from tests.data_generation import (
    _OpenSetTestData,
    _RiemannianMetricTestData,
    generate_random_vec,
)


class PoissonTestData(_OpenSetTestData):
    Space = PoissonDistributions
    n_list = random.sample((2, 5), 1)
    n_samples_list = random.sample(range(1, 10), 3)
    space_args_list = []
    metric_args_list = []
    shape_list = [(1,)]
    n_points_list = random.sample(range(1, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 1)

    def belongs_test_data(self):
        smoke_data = [
            dict(point=gs.array([5.0]), expected=True),
            dict(point=gs.array([-2.0]), expected=False),
            dict(point=gs.array([[1.0], [-1.0]]), expected=gs.array([True, False])),
            dict(point=gs.array([[0.1], [10]]), expected=gs.array([True, True])),
            dict(point=gs.array([[-0.1], [-10]]), expected=gs.array([False, False])),
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
        ]
        return self.generate_tests([], random_data)


class PoissonMetricTestData(_RiemannianMetricTestData):
    Metric = PoissonMetric

    connection_args_list = metric_args_list = [{}]
    space_list = [PoissonDistributions()]
    shape_list = [(1,)]

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
                space=self.space_list[0],
                point_a=gs.array([1, 3, 0.1]),
                point_b=gs.array([4, 3, 0.9]),
                expected=gs.array([4.0, 0.0, 1.6]),
            ),
            dict(
                space=self.space_list[0],
                point_a=gs.array(0.1),
                point_b=gs.array(4.9),
                expected=gs.array(14.4),
            ),
            dict(
                space=self.space_list[0],
                point_a=gs.array(0.1),
                point_b=gs.array([4.9, 0.9]),
                expected=gs.array([14.4, 1.6]),
            ),
            dict(
                space=self.space_list[0],
                point_a=gs.array([4.9, 0.4]),
                point_b=gs.array(0.1),
                expected=gs.array([14.4, 0.4]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def metric_matrix_test_data(self):
        smoke_data = [
            dict(
                space=self.space_list[0],
                point=gs.array([0.5]),
                expected=gs.array([[2]]),
            ),
            dict(
                space=self.space_list[0],
                point=gs.array([[0.5], [0.2]]),
                expected=gs.array([[[2]], [[5]]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def geodesic_symmetry_test_data(self):
        random_data = []
        for space in self.space_list:
            random_data.append(dict(space=space))
        return self.generate_tests([], random_data)

    def log_after_exp_test_data(self):
        random_data = []
        for connection_args, space, shape, n_tangent_vecs in zip(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        ):
            base_point = space.random_point()
            base_point_type = base_point.dtype
            random_vec = generate_random_vec(
                shape=(n_tangent_vecs,) + shape, dtype=base_point_type
            )
            random_vec = gs.where(random_vec > 0, random_vec, -random_vec)
            tangent_vec = space.to_tangent(random_vec, base_point)
            random_data.append(
                dict(
                    space=space,
                    connection_args=connection_args,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)
