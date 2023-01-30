import random

import geomstats.backend as gs
from geomstats.information_geometry.geometric import (
    GeometricDistributions,
    GeometricMetric,
)
from tests.data_generation import (
    _OpenSetTestData,
    _RiemannianMetricTestData,
    generate_random_vec,
)


class GeometricTestData(_OpenSetTestData):
    Space = GeometricDistributions
    n_list = random.sample((2, 5), 1)
    n_samples_list = random.sample(range(1, 10), 3)
    space_args_list = []
    metric_args_list = []
    shape_list = [(1,)]
    n_points_list = random.sample(range(1, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 1)

    def belongs_test_data(self):
        smoke_data = [
            dict(point=gs.array([0.1]), expected=True),
            dict(point=gs.array([5.0]), expected=False),
            dict(point=gs.array([-2.0]), expected=False),
            dict(point=gs.array([[0.9], [-1.0]]), expected=gs.array([True, False])),
            dict(point=gs.array([[0.1], [0.7]]), expected=gs.array([True, True])),
            dict(point=gs.array([[-0.1], [3.7]]), expected=gs.array([False, False])),
        ]
        return self.generate_tests(smoke_data)

    def random_point_shape_test_data(self):
        random_data = [
            dict(point=self.Space().random_point(3), expected=(3, 1)),
            dict(point=self.Space().random_point(2), expected=(2, 1)),
            dict(point=self.Space().random_point(1), expected=(1,)),
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
                point=self.Space().random_point(2),
                n_samples=3,
            ),
        ]
        return self.generate_tests([], random_data)


class GeometricMetricTestData(_RiemannianMetricTestData):
    Space = GeometricDistributions
    Metric = GeometricMetric

    n_list = random.sample((2, 5), 2)
    n_samples_list = random.sample(range(1, 10), 3)
    connection_args_list = metric_args_list = [() for n in n_list]
    space_list = [GeometricDistributions() for n in n_list]
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
                point_a=gs.array([0.2, 0.3]),
                point_b=gs.array([0.3, 0.5]),
                expected=gs.array([0.21846342154512002, 0.4318107273293949]),
            ),
            dict(
                point_a=gs.array(0.2),
                point_b=gs.array(0.3),
                expected=gs.array(0.21846342154512002),
            ),
            dict(
                point_a=gs.array(0.3),
                point_b=gs.array([0.2, 0.5]),
                expected=gs.array([0.21846342154512002, 0.4318107273293949]),
            ),
            dict(
                point_a=gs.array([0.2, 0.5]),
                point_b=gs.array(0.3),
                expected=gs.array([0.21846342154512002, 0.4318107273293949]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def metric_matrix_test_data(self):
        smoke_data = [
            dict(
                point=gs.array([0.5]),
                expected=gs.array([[8.0]]),
            ),
            dict(
                point=gs.array([[0.2], [0.5], [0.4]]),
                expected=gs.array(
                    [[[31.249999999999993]], [[8.0]], [[10.416666666666664]]]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)

    def geodesic_symmetry_test_data(self):
        random_data = []
        for space_args in self.space_args_list:
            random_data.append(dict(space_args=space_args))
        return self.generate_tests([], random_data)

    def exp_belongs_test_data(self):
        """Redefine data generation for test_exp_belongs by limiting tangent vector
        to a specific range so that results of exp are not too close to zero.
        """
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
            max_tangent_vec = (
                (gs.arctanh(1e-3) - gs.arctanh(gs.sqrt(1 - base_point)))
                * 2
                * base_point
                * gs.sqrt(1 - base_point)
            )
            min_tangent_vec = (
                -(gs.arctanh(gs.sqrt(1 - 1e-6)) - gs.arctanh(gs.sqrt(1 - base_point)))
                * 2
                * base_point
                * gs.sqrt(1 - base_point)
            )
            random_vec = gs.where(
                (random_vec > min_tangent_vec) & (random_vec < max_tangent_vec),
                random_vec,
                random_vec % (max_tangent_vec - min_tangent_vec) + min_tangent_vec,
            )
            tangent_vec = space.to_tangent(random_vec, base_point)
            random_data.append(
                dict(
                    connection_args=connection_args,
                    space=space,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)

    def geodesic_ivp_belongs_test_data(self):
        """Redefine data generation for test_geodesic_ivp_belongs by limiting
        tangent vector to a specific range so that results of geodesic_ivp are
        not too close to zero.
        """
        random_data = []
        for connection_args, space, n_points, shape in zip(
            self.metric_args_list, self.space_list, self.n_points_list, self.shape_list
        ):
            initial_point = space.random_point()
            initial_point_type = initial_point.dtype
            random_vec = generate_random_vec(shape=shape, dtype=initial_point_type)
            max_tangent_vec = (
                (gs.arctanh(1e-3) - gs.arctanh(gs.sqrt(1 - initial_point)))
                * 2
                * initial_point
                * gs.sqrt(1 - initial_point)
            )
            min_tangent_vec = (
                -(
                    gs.arctanh(gs.sqrt(1 - 1e-6))
                    - gs.arctanh(gs.sqrt(1 - initial_point))
                )
                * 2
                * initial_point
                * gs.sqrt(1 - initial_point)
            )
            random_vec = gs.where(
                (random_vec > min_tangent_vec) & (random_vec < max_tangent_vec),
                random_vec,
                random_vec % (max_tangent_vec - min_tangent_vec) + min_tangent_vec,
            )
            initial_tangent_vec = space.to_tangent(random_vec, initial_point)
            random_data.append(
                dict(
                    connection_args=connection_args,
                    space=space,
                    n_points=n_points,
                    initial_point=initial_point,
                    initial_tangent_vec=initial_tangent_vec,
                )
            )
        return self.generate_tests([], random_data)

    def log_after_exp_test_data(self):
        """Redefine data generation for test_log_after_exp by limiting tangent
        vector to a specific range so that results of exp are not too close to zero.
        """
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
            max_tangent_vec = (
                (gs.arctanh(1e-3) - gs.arctanh(gs.sqrt(1 - base_point)))
                * 2
                * base_point
                * gs.sqrt(1 - base_point)
            )
            min_tangent_vec = (
                -(gs.arctanh(gs.sqrt(1 - 1e-6)) - gs.arctanh(gs.sqrt(1 - base_point)))
                * 2
                * base_point
                * gs.sqrt(1 - base_point)
            )
            random_vec = gs.where(
                (random_vec > min_tangent_vec) & (random_vec < max_tangent_vec),
                random_vec,
                random_vec % (max_tangent_vec - min_tangent_vec) + min_tangent_vec,
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
