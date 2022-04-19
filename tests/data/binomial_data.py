import random

import geomstats.backend as gs
from geomstats.information_geometry.binomial import BinomialDistributions
from tests.data_generation import _OpenSetTestData


class BinomialTestData(_OpenSetTestData):
    space = BinomialDistributions
    n_list = random.sample((2, 5), 1)
    n_samples_list = random.sample(range(10), 3)
    space_args_list = [(n,) for n in n_list]
    shape_list = [(1,)]
    n_points_list = random.sample(range(5), 2)
    n_vecs_list = random.sample(range(2, 5), 1)

    def belongs_test_data(self):
        smoke_data = [
            dict(n_draws=10, point=0.1, expected=True),
            dict(n_draws=8, point=-0.8, expected=False),
            dict(n_draws=5, point=0.1, expected=True),
            dict(n_draws=2, point=-1.0, expected=False),
            dict(n_draws=1, point=0.1, expected=True),
            dict(n_draws=7, point=gs.array(-0.8), expected=False),
            dict(n_draws=8, point=8, expected=False),
            dict(n_draws=2, point=-1.0, expected=False),
            dict(n_draws=1, point=gs.array([5]), expected=False),
            dict(n_draws=1, point=gs.array(-0.2), expected=False),
            dict(
                n_draws=3, point=gs.array([0.9, -1]), expected=gs.array([True, False])
            ),
            dict(
                n_draws=5,
                point=gs.array([[0.1], [10]]),
                expected=gs.array([True, False]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def random_point_test_data(self):
        random_data = [
            dict(point=self.space(5).random_point(3), expected=(3,)),
            dict(point=self.space(10).random_point(2), expected=(2,)),
            dict(point=self.space(3).random_point(1), expected=()),
        ]
        return self.generate_tests([], random_data)

    def random_point_belongs_test_data(self):
        smoke_space_args_list = [(7,), (10,)]
        smoke_n_points_list = [1, 2, 5]
        return self._random_point_belongs_test_data(
            smoke_space_args_list,
            smoke_n_points_list,
            self.space_args_list,
            self.n_points_list,
        )

    def projection_belongs_test_data(self):
        return self._projection_belongs_test_data(
            self.space_args_list, self.shape_list, self.n_samples_list
        )

    def to_tangent_is_tangent_test_data(self):
        return self._to_tangent_is_tangent_test_data(
            self.space,
            self.space_args_list,
            self.shape_list,
            self.n_vecs_list,
        )

    def to_tangent_is_tangent_in_ambient_space_test_data(self):
        return self._to_tangent_is_tangent_in_ambient_space_test_data(
            self.space,
            self.space_args_list,
            self.shape_list,
        )

    def random_tangent_vec_is_tangent_test_data(self):
        return self._random_tangent_vec_is_tangent_test_data(
            self.space,
            self.space_args_list,
            self.n_vecs_list,
            is_tangent_atol=gs.atol,
        )

    def sample_test_data(self):
        smoke_data = [
            dict(n_draws=5, point=gs.array([0.2, 0.3]), n_samples=2, expected=(2, 2)),
            dict(
                n_draws=10,
                point=gs.array([0.1, 0.2, 0.3]),
                n_samples=1,
                expected=(3,),
            ),
        ]
        return self.generate_tests(smoke_data)

    def sample_belongs_test_data(self):
        random_data = [
            dict(
                n_draws=2,
                point=self.space(2).random_point(3),
                n_samples=4,
                expected=gs.ones((3, 4)),
            ),
            dict(
                n_draws=3,
                point=self.space(3).random_point(1),
                n_samples=2,
                expected=gs.ones(2),
            ),
            dict(
                n_draws=4,
                point=self.space(4).random_point(2),
                n_samples=3,
                expected=gs.ones((2, 3)),
            ),
        ]
        return self.generate_tests([], random_data)

    def point_to_pdf_test_data(self):
        random_data = [
            dict(
                n_draws=5,
                point=self.space(5).random_point(2),
                n_samples=3,
            ),
            dict(
                n_draws=3,
                point=self.space(3).random_point(1),
                n_samples=4,
            ),
            dict(
                n_draws=7,
                point=self.space(7).random_point(1),
                n_samples=1,
            ),
            dict(
                n_draws=2,
                point=self.space(2).random_point(4),
                n_samples=1,
            ),
        ]
        return self.generate_tests([], random_data)

    def triangle_inequality_of_dist_test_data(self):
        return self._triangle_inequality_of_dist_test_data(
            self.metric_args_list,
            self.space_args_list,
            self.n_points_list,
            atol=gs.atol,
        )

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
        ]
        return self.generate_tests(smoke_data)
