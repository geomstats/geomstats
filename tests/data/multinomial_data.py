import random

import geomstats.backend as gs
from geomstats.information_geometry.multinomial import (
    MultinomialDistributions,
    MultinomialMetric,
)
from tests.data_generation import _LevelSetTestData, _RiemannianMetricTestData


class MultinomialTestData(_LevelSetTestData):
    Space = MultinomialDistributions
    dim_list = random.sample(range(2, 5), 2)
    n_draws_list = random.sample(range(2, 5), 2)
    space_args_list = zip(dim_list, n_draws_list)
    shape_list = [(dim + 1,) for dim in dim_list]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = n_vecs_list = random.sample(range(2, 5), 2)

    def belongs_test_data(self):
        smoke_data = [
            dict(dim=3, vec=[0.1, 0.3, 0.3, 0.4], expected=True),
            dict(dim=3, vec=[0.1, 1.0], expected=False),
            dict(dim=3, vec=[0.0, 1.0, 0.3, 0.4], expected=False),
            dict(dim=2, vec=[-1.0, 0.3], expected=False),
        ]
        return self.generate_tests(smoke_data)

    def random_point_test_data(self):
        random_data = [
            dict(point=self.Space(dim=2, n_draws=3).random_point(1), expected=(3,)),
            dict(point=self.Space(dim=3, n_draws=6).random_point(5), expected=(5, 4)),
        ]
        return self.generate_tests([], random_data)

    def sample_shape_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                n_draws=4,
                point=gs.array([0.2, 0.3, 0.5]),
                n_samples=1,
                expected=(3,),
            ),
            dict(
                dim=3,
                n_draws=2,
                point=gs.array([0.1, 0.2, 0.3, 0.4]),
                n_samples=2,
                expected=(2, 4),
            ),
        ]
        return self.generate_tests(smoke_data)


class MultinomialMetricTestData(_RiemannianMetricTestData):
    Space = MultinomialDistributions
    Metric = MultinomialMetric

    dim_list = random.sample(range(2, 5), 2)
    n_draws_list = random.sample(range(2, 5), 2)
    connection_args_list = metric_args_list = list(zip(dim_list, n_draws_list))
    space_list = [
        MultinomialDistributions(dim, n_draws)
        for dim, n_draws in zip(dim_list, n_draws_list)
    ]
    space_args_list = zip(dim_list, n_draws_list)
    n_samples_list = random.sample(range(2, 5), 2)
    shape_list = [(dim + 1,) for dim in dim_list]
    n_points_a_list = n_points_b_list = n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = n_vecs_list = random.sample(range(2, 5), 2)

    def sectional_curvature_is_positive_test_data(self):
        random_data = [
            dict(dim=2, n_draws=3, base_point=self.Space(2, 3).random_point()),
            dict(dim=3, n_draws=1, base_point=self.Space(3, 1).random_point()),
            dict(dim=4, n_draws=2, base_point=self.Space(4, 2).random_point()),
        ]
        return self.generate_tests([], random_data)
