import random

import geomstats.backend as gs
from geomstats.information_geometry.binomial import (
    BinomialDistributions,
    BinomialMetric,
)
from tests.data_generation import (
    _OpenSetTestData,
    _RiemannianMetricTestData,
    generate_random_vec,
)


class BinomialTestData(_OpenSetTestData):
    Space = BinomialDistributions
    n_list = random.sample((2, 5), 1)
    n_samples_list = random.sample(range(1, 10), 3)
    space_args_list = [(n,) for n in n_list]
    shape_list = [(1,)]
    n_points_list = random.sample(range(1, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 1)

    def belongs_test_data(self):
        smoke_data = [
            dict(n_draws=10, point=gs.array([0.1]), expected=True),
            dict(n_draws=8, point=gs.array([-0.8]), expected=False),
            dict(n_draws=5, point=gs.array([0.1]), expected=True),
            dict(n_draws=2, point=gs.array([-1.0]), expected=False),
            dict(n_draws=1, point=gs.array([0.1]), expected=True),
            dict(n_draws=7, point=gs.array([-0.8]), expected=False),
            dict(n_draws=8, point=gs.array([8.0]), expected=False),
            dict(n_draws=2, point=gs.array([-1.0]), expected=False),
            dict(n_draws=1, point=gs.array([5.0]), expected=False),
            dict(n_draws=1, point=gs.array([-0.2]), expected=False),
            dict(
                n_draws=3,
                point=gs.array([[0.9], [-1]]),
                expected=gs.array([True, False]),
            ),
            dict(
                n_draws=5,
                point=gs.array([[0.1], [10]]),
                expected=gs.array([True, False]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def random_point_shape_test_data(self):
        random_data = [
            dict(point=self.Space(5).random_point(3), expected=(3, 1)),
            dict(point=self.Space(10).random_point(2), expected=(2, 1)),
            dict(point=self.Space(3).random_point(1), expected=(1,)),
        ]
        return self.generate_tests([], random_data)

    def sample_shape_test_data(self):
        smoke_data = [
            dict(
                n_draws=5, point=gs.array([[0.2], [0.3]]), n_samples=2, expected=(2, 2)
            ),
            dict(
                n_draws=10,
                point=gs.array([[0.1], [0.2], [0.3]]),
                n_samples=1,
                expected=(3, 1),
            ),
        ]
        return self.generate_tests(smoke_data)

    def sample_belongs_test_data(self):
        random_data = [
            dict(
                n_draws=2,
                point=self.Space(2).random_point(3),
                n_samples=4,
                expected=gs.ones((3, 4)),
            ),
            dict(
                n_draws=3,
                point=self.Space(3).random_point(1),
                n_samples=2,
                expected=gs.ones(2),
            ),
            dict(
                n_draws=4,
                point=self.Space(4).random_point(2),
                n_samples=3,
                expected=gs.ones((2, 3)),
            ),
        ]
        return self.generate_tests([], random_data)

    def point_to_pdf_test_data(self):
        random_data = [
            dict(
                n_draws=3,
                point=self.Space(3).random_point(1),
                n_samples=4,
            ),
            dict(
                n_draws=7,
                point=self.Space(7).random_point(1),
                n_samples=1,
            ),
            dict(
                n_draws=2,
                point=self.Space(2).random_point(4),
                n_samples=1,
            ),
        ]
        return self.generate_tests([], random_data)


class BinomialMetricTestData(_RiemannianMetricTestData):
    Space = BinomialDistributions
    Metric = BinomialMetric

    n_list = random.sample((2, 5), 2)
    n_samples_list = random.sample(range(1, 10), 3)
    connection_args_list = metric_args_list = [(n,) for n in n_list]
    space_list = [BinomialDistributions(n) for n in n_list]
    space_args_list = [(n,) for n in n_list]
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
                n_draws=5,
                point_a=gs.array([0.2, 0.3]),
                point_b=gs.array([0.3, 0.5]),
                expected=gs.array([0.26908349, 0.84673057]),
            ),
            dict(
                n_draws=10,
                point_a=gs.array(0.1),
                point_b=gs.array(0.99),
                expected=gs.array(52.79685863761384),
            ),
            dict(
                n_draws=5,
                point_a=gs.array(0.3),
                point_b=gs.array([0.2, 0.5]),
                expected=gs.array([0.26908349, 0.84673057]),
            ),
            dict(
                n_draws=5,
                point_a=gs.array([0.2, 0.5]),
                point_b=gs.array(0.3),
                expected=gs.array([0.26908349, 0.84673057]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def metric_matrix_test_data(self):
        smoke_data = [
            dict(
                n_draws=5,
                point=gs.array([0.5]),
                expected=gs.array([[20.0]]),
            ),
            dict(
                n_draws=7,
                point=gs.array([[0.1], [0.5], [0.4]]),
                expected=gs.array(
                    [[[77.77777777777777]], [[28.0]], [[29.166666666666668]]]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)

    def geodesic_symmetry_test_data(self):
        random_data = []
        for space_args in self.space_args_list:
            random_data.append(dict(space_args=space_args))
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
            random_vec = random_vec % (
                gs.pi * gs.sqrt(base_point * (1 - base_point))
            ) + (-gs.arcsin(gs.sqrt(base_point))) * 2 * gs.sqrt(
                base_point * (1 - base_point)
            )
            tangent_vec = space.to_tangent(random_vec, base_point)
            random_data.append(
                dict(
                    connection_args=connection_args,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)
