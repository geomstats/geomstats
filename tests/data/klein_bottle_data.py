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

    def not_belongs_test_data(self):
        random_data = [
            dict(point=gs.random.rand(3, 4, 2) + 1, expected=gs.zeros((3, 4)))
        ]
        return self.generate_tests([], random_data)

    def is_tangent_wrong_shape_test_data(self):
        return self.not_belongs_error_test_data()

    def to_tangent_wrong_shape_test_data(self):
        return self.not_belongs_error_test_data()

    def not_belongs_error_test_data(self):
        space = self.Space()
        point_small = gs.array(42)
        point_large = gs.random.rand(8, *space.shape, 3)
        random_data = [
            dict(
                point=point_small,
            ),
            dict(
                point=point_large,
            ),
        ]
        return self.generate_tests([], random_data)

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
            dict(
                regularized=gs.array([[0.0, 0.0], [0.0, 0.0]]),
                point=gs.array([[1.0, 1.0], [-1.0, -1.0]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def regularize_correct_domain_test_data(self):
        space = self.Space()
        smoke_data = [dict(points=gs.array([[1.0, 1.0], [-1.0, -1.0], [2.0, -10.0]]))]
        random_data = [dict(points=space.random_point(10))]
        return self.generate_tests(smoke_data, random_data)


class KleinBottleMetricTestData(_RiemannianMetricTestData):
    n_points_list = random.sample(range(1, 5), 2)

    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = n_points_a_list

    space_list = [KleinBottle(), KleinBottle()]
    space = KleinBottle()
    connection_args_list = metric_args_list = [[space] for space in space_list]
    shape_list = [(2,), (2,)]
    n_tangent_vecs_list = random.sample(range(1, 5), 2)

    Metric = KleinBottleMetric

    def log_after_exp_test_data(self, **kwargs):
        """Generate data to check that exponential and logarithm are inverse.

        Use random tangent vectors with length
        bounded by injectivity radius of the manifold equipped with the tested metric.
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
            scale = (
                gs.random.uniform(
                    size=(n_tangent_vecs,) + (1,) * (len(random_vec.shape) - 1)
                )
                * inj_radius
            )
            random_vec *= scale
            random_data.append(
                dict(
                    connection_args=connection_args,
                    tangent_vec=random_vec,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)

    def dist_test_data(self):
        smoke_data = [
            dict(
                point_a=gs.array([0.5, 0.5]),
                point_b=gs.array([0.0, 0.0]),
                expected=gs.array([2**0.5 / 2]),
            ),
            dict(
                point_a=gs.array([0.1, 0.12]),
                point_b=gs.array([0.9, 0.8]),
                expected=gs.array([(0.2**2 + (0.2 - 0.12) ** 2) ** 0.5]),
            ),
            dict(
                point_a=gs.array([0.2, 0.8]),
                point_b=gs.array([0.8, 0.8]),
                expected=gs.array([(0.4**2 + 0.4**2) ** 0.5]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def diameter_test_data(self):
        points = self.space.random_point(10)
        point1 = gs.array([[0.0, 0.0]])
        point2 = gs.array([[0.5, 0.5]])
        points = gs.concatenate([points, point1, point2])
        smoke_data = [dict(points=points, expected=2**0.5 / 2)]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                base_point=gs.array([0.6, 0.3]),
                tangent_vec=gs.array(
                    [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [2.0, 0.2], [-0.1, 1.1]]
                ),
                expected=gs.array(
                    [[0.6, 0.7], [0.6, 0.3], [0.6, 0.3], [0.6, 0.5], [0.5, 0.4]]
                ),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                base_point=gs.array([0.6, 0.3]),
                point=gs.array([[0.6, 0.7], [0.6, 0.3], [0.6, 0.5]]),
                expected=gs.array([[0.0, 0.4], [0.0, 0.0], [0.0, 0.2]]),
            ),
            dict(
                base_point=gs.array([0.1, 0.12]),
                point=gs.array([0.9, 0.8]),
                expected=gs.array([-0.1 - 0.1, 0.2 - 0.12]),
            ),
        ]
        return self.generate_tests(smoke_data)
