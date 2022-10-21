import random

import geomstats.backend as gs
from geomstats.information_geometry.exponential import ExponentialDistributions
from tests.data_generation import _OpenSetTestData


class ExponentialTestData(_OpenSetTestData):
    Space = ExponentialDistributions
    n_list = random.sample((2, 5), 1)
    n_samples_list = random.sample(range(10), 3)
    space_args_list = []
    metric_args_list = []
    shape_list = [(1,)]
    n_points_list = random.sample(range(5), 2)
    n_vecs_list = random.sample(range(2, 5), 1)
    batch_shape_list = [
        tuple(random.choices(range(2, 10), k=i)) for i in random.sample(range(1, 5), 3)
    ]

    def belongs_test_data(self):
        smoke_data = [
            dict(point=0.1, expected=True),
            dict(point=gs.array(-0.8), expected=False),
            dict(point=8, expected=True),
            dict(point=-1.0, expected=False),
            dict(point=gs.array([5.0]), expected=True),
            dict(point=gs.array(-2.0), expected=False),
            dict(point=gs.array([1.0, -1.0]), expected=gs.array([True, False])),
            dict(point=gs.array([[0.1], [10]]), expected=gs.array([True, True])),
        ]
        return self.generate_tests(smoke_data)

    def random_point_test_data(self):
        random_data = [
            dict(point=self.Space().random_point(3), expected=(3,)),
            dict(point=self.Space().random_point(2), expected=(2,)),
            dict(point=self.Space().random_point(1), expected=()),
        ]
        return self.generate_tests([], random_data)

    def sample_test_data(self):
        smoke_data = [
            dict(point=gs.array([0.2, 0.3]), n_samples=2, expected=(2, 2)),
            dict(
                point=gs.array([0.1, 0.2, 0.3]),
                n_samples=1,
                expected=(3,),
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
                point=self.Space().random_point(2),
                n_samples=3,
            ),
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
        ]
        return self.generate_tests(smoke_data)
