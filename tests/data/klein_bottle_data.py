import random

import geomstats.backend as gs
from geomstats.geometry.klein_bottle import KleinBottle, KleinBottleMetric
from tests.data_generation import _ManifoldTestData, _RiemannianMetricTestData


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
                regularized=gs.array([[0.1, 0.1], [0.5, 0.6], [0.9, 0.4]]),
                point=gs.array([[1.1, -0.1], [-0.5, 0.4], [0.9, 0.4]]),
            ),
        ]
        return self.generate_tests(smoke_data)


class KleinBottleMetricTestData(_RiemannianMetricTestData):
    import numpy as np
    np.random.seed(42)
    n_points_list = random.sample(range(1, 5), 2)

    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = n_points_a_list

    space_list = [KleinBottle(), KleinBottle()]
    connection_args_list = metric_args_list = [[space] for space in space_list]
    shape_list = [(2,), (2,)]
    n_tangent_vecs_list = random.sample(range(1, 5), 2)

    # Space = KleinBottle
    Metric = KleinBottleMetric

    def log_after_exp_test_data(self, **kwargs):
        """Generate data to check that exponential and logarithm are inverse.
        """
        random_data = []
        for connection_args, space, n_tangent_vecs in zip(
            self.metric_args_list,
            self.space_list,
            self.n_tangent_vecs_list,
        ):
            connection = self.Metric(*connection_args)
            base_point = space.random_point()
            random_vec = space.random_tangent_vec(base_point, n_tangent_vecs)
            random_vec = connection.normalize(random_vec, base_point)
            inj_radius = connection.injectivity_radius(base_point)
            scale = gs.random.uniform(size=(n_tangent_vecs,) + (1,)*(len(random_vec.shape)-1)) * inj_radius
            random_vec *= scale
            random_data.append(
                dict(
                    connection_args=connection_args,
                    tangent_vec=random_vec,
                    base_point=base_point,
                )
            )
        smoke_data = [dict(
            connection_args=self.metric_args_list[0],
            tangent_vec=gs.array([0.5, -0.4]),
            base_point=gs.array([0.6, 0.8])
        )]
        return self.generate_tests(smoke_data, random_data)

