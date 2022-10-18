import random

import geomstats.backend as gs
from geomstats.geometry.klein_bottle import KleinBottle
from tests.data_generation import _ManifoldTestData


class KleinBottleTestData(_ManifoldTestData):
    dim_list = [2, 2]
    space_args_list = [(dim,) for dim in dim_list]
    n_points_list = random.sample(range(2, 5), 2)
    shape_list = [(dim,) for dim in dim_list]
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = KleinBottle

    def equivalent_test_data(self):
        smoke_data = [
            dict(
                point1=gs.array([0.3, 0.7]),
                point2=gs.array([2.3, 0.7]),
                expected=gs.array(True),
            ),
            dict(
                point1=gs.array([0.45 - 2, 0.67]),
                point2=gs.array([1.45, 1 - 0.67]),
                expected=gs.array(True),
            ),
            dict(
                point1=gs.array([0.11, 0.12]),
                point2=gs.array([0.11 - 1, 1 - 0.12]),
                expected=gs.array(True),
            ),
            dict(
                point1=gs.array([0.1, 0.12]),
                point2=gs.array([0.1 + 2 + gs.atol / 2, 0.12]),
                expected=gs.array(True),
            ),
            dict(
                point1=gs.array([0.1, 0.12]),
                point2=gs.array([0.1 + 2 - gs.atol / 2, 0.12]),
                expected=gs.array(True),
            ),
            dict(
                point1=gs.array([[0.1, 0.1], [0.5, 0.4]]),
                point2=gs.array([[1.1, -0.1], [-0.5, 0.4]]),
                expected=gs.array([True, False]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def regularize_test_data(self):
        smoke_data = [
            dict(regularized=gs.array([0.3, 0.7]), point=gs.array([2.3, 0.7])),
            dict(regularized=gs.array([0.45, 0.67]), point=gs.array([1.45, 1 - 0.67])),
            dict(
                regularized=gs.array([0.11, 0.12]), point=gs.array([0.11 - 1, 1 - 0.12])
            ),
            dict(
                regularized=gs.array([gs.atol / 3 + gs.atol / 2, 0.12]),
                point=gs.array([gs.atol / 3 + 2 + gs.atol / 2, 0.12]),
            ),
            dict(
                regularized=gs.array([gs.atol / 3 - gs.atol / 2 + 1, 1 - 0.12]),
                point=gs.array([gs.atol / 3 + 2 - gs.atol / 2, 0.12]),
            ),
            dict(
                regularized=gs.array([[0.1, 0.1], [0.5, 0.6]]),
                point=gs.array([[1.1, -0.1], [-0.5, 0.4]]),
            ),
        ]
        return self.generate_tests(smoke_data)
