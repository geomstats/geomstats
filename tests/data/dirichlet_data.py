import random

import geomstats.backend as gs
from geomstats.information_geometry.dirichlet import (
    DirichletDistributions,
    DirichletMetric,
)
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData


class DirichletTestData(_OpenSetTestData):
    Space = DirichletDistributions
    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    shape_list = [(n,) for n in n_list]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = n_vecs_list = random.sample(range(2, 5), 2)

    def belongs_test_data(self):
        smoke_data = [
            dict(dim=3, vec=gs.array([0.1, 1.0, 0.3]), expected=True),
            dict(dim=3, vec=gs.array([0.1, 1.0]), expected=False),
            dict(dim=3, vec=gs.array([0.0, 1.0, 0.3]), expected=False),
            dict(dim=2, vec=gs.array([-1.0, 0.3]), expected=False),
        ]
        return self.generate_tests(smoke_data)

    def random_point_test_data(self):
        random_data = [
            dict(point=self.Space(2).random_point(1), expected=(2,)),
            dict(point=self.Space(3).random_point(5), expected=(5, 3)),
        ]
        return self.generate_tests([], random_data)

    def sample_test_data(self):
        smoke_data = [
            dict(dim=2, point=gs.array([1.0, 1.0]), n_samples=1, expected=(1, 2)),
            dict(dim=3, point=gs.array([0.1, 0.2, 0.3]), n_samples=2, expected=(2, 3)),
        ]
        return self.generate_tests(smoke_data)

    def sample_belongs_test_data(self):
        random_data = [
            dict(
                dim=2,
                point=self.Space(2).random_point(3),
                n_samples=4,
                expected=gs.ones((3, 4)),
            ),
            dict(
                dim=3,
                point=self.Space(3).random_point(1),
                n_samples=2,
                expected=gs.ones(2),
            ),
            dict(
                dim=4,
                point=self.Space(4).random_point(2),
                n_samples=3,
                expected=gs.ones((2, 3)),
            ),
        ]
        return self.generate_tests([], random_data)

    def point_to_pdf_test_data(self):
        random_data = [
            dict(
                dim=2,
                point=self.Space(2).random_point(2),
                n_samples=10,
            ),
            dict(
                dim=3,
                point=self.Space(3).random_point(4),
                n_samples=10,
            ),
            dict(
                dim=4,
                point=self.Space(4).random_point(1),
                n_samples=10,
            ),
        ]
        return self.generate_tests([], random_data)


class DirichletMetricTestData(_RiemannianMetricTestData):
    Space = DirichletDistributions
    Metric = DirichletMetric

    dim_list = random.sample(range(2, 4), 2)
    connection_args_list = metric_args_list = [{} for _ in dim_list]

    space_list = [DirichletDistributions(n) for n in dim_list]
    shape_list = [(n,) for n in dim_list]

    n_points_a_list = n_points_b_list = n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = n_vecs_list = random.sample(range(2, 5), 2)

    tolerances = {
        "dist_point_to_itself_is_zero": {"atol": 1e-5},
        "dist_is_symmetric": {"atol": 5e-1},
        "dist_is_norm_of_log": {"atol": 5e-1},
        "exp_subspace": {"atol": 1e-4},
        "triangle_inequality_of_dist": {"atol": 1e-10},
    }

    def metric_matrix_shape_test_data(self):
        random_data = [
            dict(space=self.Space(2), n_points=1, expected=(2, 2)),
            dict(space=self.Space(2), n_points=3, expected=(3, 2, 2)),
            dict(space=self.Space(3), n_points=2, expected=(2, 3, 3)),
        ]
        return self.generate_tests([], random_data)

    def metric_matrix_dim_2_test_data(self):
        random_data = [
            dict(space=self.Space(2), n_points=n_points)
            for n_points in self.n_points_list
        ]
        return self.generate_tests([], random_data)

    def christoffels_vectorization_test_data(self):
        random_data = [dict(space=space) for space in self.space_list]
        return self.generate_tests([], random_data)

    def christoffels_shape_test_data(self):
        random_data = [
            dict(space=self.Space(2), n_points=1, expected=(2, 2, 2)),
            dict(space=self.Space(2), n_points=3, expected=(3, 2, 2, 2)),
            dict(space=self.Space(3), n_points=2, expected=(2, 3, 3, 3)),
        ]
        return self.generate_tests([], random_data)

    def christoffels_dim_2_test_data(self):
        random_data = [dict(space=self.Space(2))]
        return self.generate_tests([], random_data)

    def exp_vectorization_test_data(self):
        dim = 3
        tangent_vec = gs.array([1.0, 0.5, 2.0])
        n_tangent_vecs = 10
        t = gs.linspace(0.0, 1.0, n_tangent_vecs)
        tangent_vecs = gs.einsum("i,...k->...ik", t, tangent_vec)
        random_data = [dict(space=self.Space(dim), tangent_vecs=tangent_vecs)]
        return self.generate_tests([], random_data)

    def exp_diagonal_test_data(self):
        param_list = [0.8, 1.2, 2.5]
        smoke_data = [
            dict(space=space, param=param, param_list=param_list)
            for space in self.space_list
            for param in param_list
        ]
        return self.generate_tests(smoke_data)

    def exp_subspace_test_data(self):
        smoke_data = [
            dict(
                space=self.Space(3),
                point=[0.1, 0.1, 0.5],
                vec=[1.3, 1.3, 2.2],
                expected=[True, True, False],
            ),
            dict(
                space=self.Space(3),
                point=[3.5, 0.1, 3.5],
                vec=[0.8, 0.1, 0.8],
                expected=[True, False, True],
            ),
            dict(
                space=self.Space(4),
                point=[1.1, 1.1, 2.3, 1.1],
                vec=[0.6, 0.6, 2.1, 0.6],
                expected=[True, True, False, True],
            ),
        ]
        return self.generate_tests(smoke_data)

    def geodesic_ivp_shape_test_data(self):
        random_data = [
            dict(
                space=self.Space(2),
                n_points=1,
                n_steps=50,
                expected=(50, 2),
            ),
            dict(
                space=self.Space(2),
                n_points=3,
                n_steps=50,
                expected=(3, 50, 2),
            ),
            dict(
                space=self.Space(3),
                n_points=4,
                n_steps=50,
                expected=(4, 50, 3),
            ),
        ]
        return self.generate_tests([], random_data)

    def geodesic_bvp_shape_test_data(self):
        random_data = [
            dict(
                space=self.Space(2),
                n_points=1,
                n_steps=50,
                expected=(50, 2),
            ),
            dict(
                space=self.Space(2),
                n_points=3,
                n_steps=50,
                expected=(3, 50, 2),
            ),
            dict(
                space=self.Space(3),
                n_points=4,
                n_steps=50,
                expected=(4, 50, 3),
            ),
        ]
        return self.generate_tests([], random_data)

    def geodesic_test_data(self):
        random_data = [dict(space=space) for space in self.space_list]
        return self.generate_tests([], random_data)

    def geodesic_shape_test_data(self):
        random_data = [
            dict(
                space=self.Space(2),
                n_points=1,
                time=0.5,
                expected=(2,),
            ),
            dict(
                space=self.Space(3),
                n_points=4,
                time=0.5,
                expected=(4, 3),
            ),
            dict(
                space=self.Space(3),
                n_points=4,
                time=gs.linspace(0.0, 1.0, 10),
                expected=(4, 10, 3),
            ),
        ]
        return self.generate_tests([], random_data)

    def jacobian_christoffels_test_data(self):
        random_data = [dict(space=space, n_points=2) for space in self.space_list]
        return self.generate_tests([], random_data)

    def jacobian_in_geodesic_bvp_test_data(self):
        random_data = [dict(space=space) for space in self.space_list]
        return self.generate_tests([], random_data)

    def approx_geodesic_bvp_test_data(self):
        random_data = [dict(space=space) for space in self.space_list]
        return self.generate_tests([], random_data)

    def polynomial_init_test_data(self):
        smoke_data = [
            dict(
                space=self.Space(3),
                point_a=[100.0, 1.0, 1.0],
                point_b=[1.0, 1.0, 100.0],
                expected=8.5,
            ),
        ]
        return self.generate_tests(smoke_data)

    def sectional_curvature_is_negative_test_data(self):
        random_data = [dict(space=space) for space in self.space_list]
        return self.generate_tests([], random_data)
