import random
from contextlib import nullcontext as does_not_raise

import pytest

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from tests.data_generation import _LevelSetTestData, _RiemannianMetricTestData


class HypersphereTestData(_LevelSetTestData):

    dim_list = random.sample(range(1, 4), 2)
    space_args_list = [(dim,) for dim in dim_list]
    n_points_list = random.sample(range(1, 5), 2)
    shape_list = [(dim + 1,) for dim in dim_list]
    n_vecs_list = random.sample(range(1, 5), 2)

    space = Hypersphere

    def replace_values_test_data(self):
        smoke_data = [
            dict(
                dim=4,
                points=gs.ones((3, 5)),
                new_points=gs.zeros((2, 5)),
                indcs=[True, False, True],
                expected=gs.stack([gs.zeros(5), gs.ones(5), gs.zeros(5)]),
            )
        ]
        return self.generate_tests(smoke_data)

    def angle_to_extrinsic_test_data(self):
        smoke_data = [
            dict(dim=1, point=gs.pi / 4, expected=gs.array([1.0, 1.0]) / gs.sqrt(2.0)),
            dict(
                dim=1,
                point=gs.array([1.0 / 3, 0.0]) * gs.pi,
                expected=gs.array([[1.0 / 2, gs.sqrt(3.0) / 2], [1.0, 0.0]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def extrinsic_to_angle_test_data(self):
        smoke_data = [
            dict(dim=1, point=gs.array([1.0, 1.0]) / gs.sqrt(2.0), expected=gs.pi / 4),
            dict(
                dim=1,
                point=gs.array([[1.0 / 2, gs.sqrt(3.0) / 2], [1.0, 0.0]]),
                expected=gs.array([1.0 / 3, 0.0]) * gs.pi,
            ),
        ]
        return self.generate_tests(smoke_data)

    def spherical_to_extrinsic_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point=gs.array([gs.pi / 2, 0]),
                expected=gs.array([1.0, 0.0, 0.0]),
            ),
            dict(
                dim=2,
                point=gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]]),
                expected=gs.array(
                    [
                        [1.0, 0.0, 0.0],
                        [
                            gs.sqrt(2.0) / 4.0,
                            gs.sqrt(2.0) / 4.0,
                            gs.sqrt(3.0) / 2.0,
                        ],
                    ]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)

    def extrinsic_to_spherical_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point=gs.array([1.0, 0.0, 0.0]),
                expected=gs.array([gs.pi / 2, 0]),
            ),
            dict(
                dim=2,
                point=gs.array(
                    [
                        [1.0, 0.0, 0.0],
                        [
                            gs.sqrt(2.0) / 4.0,
                            gs.sqrt(2.0) / 4.0,
                            gs.sqrt(3.0) / 2.0,
                        ],
                    ]
                ),
                expected=gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def random_von_mises_fisher_belongs_test_data(self):
        dim_list = random.sample(range(2, 8), 5)
        n_samples_list = random.sample(range(1, 10), 5)
        random_data = [
            dict(dim=dim, n_samples=n_samples)
            for dim, n_samples in zip(dim_list, n_samples_list)
        ]
        return self.generate_tests([], random_data)

    def random_von_mises_fisher_mean_test_data(self):
        dim_list = random.sample(range(2, 8), 5)
        smoke_data = [
            dict(
                dim=dim,
                kappa=10,
                n_samples=100000,
                expected=gs.array([1.0] + [0.0] * dim),
            )
            for dim in dim_list
        ]
        return self.generate_tests(smoke_data)

    def tangent_extrinsic_to_spherical_raises_test_data(self):
        smoke_data = []
        dim_list = [2, 3]
        for dim in dim_list:
            space = self.space(dim)
            base_point = space.random_point()
            tangent_vec = space.to_tangent(space.random_point(), base_point)
            if dim == 2:
                expected = does_not_raise()
                smoke_data.append(
                    dict(
                        dim=2,
                        tangent_vec=tangent_vec,
                        base_point=None,
                        base_point_spherical=None,
                        expected=pytest.raises(ValueError),
                    )
                )
            else:
                expected = pytest.raises(NotImplementedError)
            smoke_data.append(
                dict(
                    dim=dim,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    base_point_spherical=None,
                    expected=expected,
                )
            )

        return self.generate_tests(smoke_data)

    def tangent_spherical_to_extrinsic_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                tangent_vec_spherical=gs.array([[0.25, 0.5], [0.3, 0.2]]),
                base_point_spherical=gs.array([[gs.pi / 2, 0], [gs.pi / 2, 0]]),
                expected=gs.array([[0, 0.5, -0.25], [0, 0.2, -0.3]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def tangent_extrinsic_to_spherical_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                tangent_vec=gs.array([[0, 0.5, -0.25], [0, 0.2, -0.3]]),
                base_point=None,
                base_point_spherical=gs.array([[gs.pi / 2, 0], [gs.pi / 2, 0]]),
                expected=gs.array([[0.25, 0.5], [0.3, 0.2]]),
            ),
            dict(
                dim=2,
                tangent_vec=gs.array([0, 0.5, -0.25]),
                base_point=gs.array([1.0, 0.0, 0.0]),
                base_point_spherical=None,
                expected=gs.array([0.25, 0.5]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def riemannian_normal_frechet_mean_test_data(self):
        smoke_data = [dict(dim=3), dict(dim=4)]
        return self.generate_tests(smoke_data)

    def riemannian_normal_and_belongs_test_data(self):
        smoke_data = [dict(dim=3, n_points=1), dict(dim=4, n_points=10)]
        return self.generate_tests(smoke_data)

    def sample_von_mises_fisher_mean_test_data(self):
        dim_list = random.sample(range(2, 10), 5)
        smoke_data = [
            dict(
                dim=dim,
                mean=self.space(dim).random_point(),
                kappa=1000.0,
                n_points=10000,
            )
            for dim in dim_list
        ]
        return self.generate_tests(smoke_data)

    def sample_random_von_mises_fisher_kappa_test_data(self):
        dim_list = random.sample(range(2, 8), 5)
        smoke_data = [dict(dim=dim, kappa=1.0, n_points=50000) for dim in dim_list]
        return self.generate_tests(smoke_data)

    def intrinsic_after_extrinsic_test_data(self):
        space_args_list = self.space_args_list
        self.space_args_list = [(1,), (2,)]

        test_data = super().intrinsic_after_extrinsic_test_data()

        self.space_args_list = space_args_list
        return test_data

    def extrinsic_after_intrinsic_test_data(self):
        space_args_list = self.space_args_list
        self.space_args_list = [(1,), (2,)]

        test_data = super().extrinsic_after_intrinsic_test_data()

        self.space_args_list = space_args_list
        return test_data


class HypersphereMetricTestData(_RiemannianMetricTestData):
    dim_list = random.sample(range(2, 5), 2)
    metric_args_list = [(n,) for n in dim_list]
    shape_list = [(dim + 1,) for dim in dim_list]
    space_list = [Hypersphere(n) for n in dim_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = HypersphereMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                dim=4,
                tangent_vec_a=[1.0, 0.0, 0.0, 0.0, 0.0],
                tangent_vec_b=[0.0, 1.0, 0.0, 0.0, 0.0],
                base_point=[0.0, 0.0, 0.0, 0.0, 1.0],
                expected=0.0,
            )
        ]
        return self.generate_tests(smoke_data)

    def dist_test_data(self):
        # smoke data is currently testing points at orthogonal
        point_a = gs.array([10.0, -2.0, -0.5, 0.0, 0.0])
        point_a = point_a / gs.linalg.norm(point_a)
        point_b = gs.array([2.0, 10, 0.0, 0.0, 0.0])
        point_b = point_b / gs.linalg.norm(point_b)
        smoke_data = [dict(dim=4, point_a=point_a, point_b=point_b, expected=gs.pi / 2)]
        return self.generate_tests(smoke_data)

    def diameter_test_data(self):
        point_a = gs.array([[0.0, 0.0, 1.0]])
        point_b = gs.array([[1.0, 0.0, 0.0]])
        point_c = gs.array([[0.0, 0.0, -1.0]])
        smoke_data = [
            dict(dim=2, points=gs.vstack((point_a, point_b, point_c)), expected=gs.pi)
        ]
        return self.generate_tests(smoke_data)

    def christoffels_shape_test_data(self):
        point = gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]])
        smoke_data = [dict(dim=2, point=point, expected=[2, 2, 2, 2])]
        return self.generate_tests(smoke_data)

    def sectional_curvature_test_data(self):
        dim_list = [4]
        n_samples_list = random.sample(range(1, 4), 2)
        random_data = []
        for dim, n_samples in zip(dim_list, n_samples_list):
            sphere = Hypersphere(dim)
            base_point = sphere.random_uniform()
            tangent_vec_a = sphere.to_tangent(
                gs.random.rand(n_samples, sphere.dim + 1), base_point
            )
            tangent_vec_b = sphere.to_tangent(
                gs.random.rand(n_samples, sphere.dim + 1), base_point
            )
            expected = gs.ones(n_samples)  # try shape here
            random_data.append(
                dict(
                    dim=dim,
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                ),
            )
        return self.generate_tests(random_data)

    def dist_pairwise_test_data(self):
        smoke_data = [
            dict(
                dim=4,
                point=[
                    1.0 / gs.sqrt(129.0) * gs.array([10.0, -2.0, -5.0, 0.0, 0.0]),
                    1.0 / gs.sqrt(435.0) * gs.array([1.0, -20.0, -5.0, 0.0, 3.0]),
                ],
                expected=gs.array([[0.0, 1.24864502], [1.24864502, 0.0]]),
                rtol=1e-3,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_after_log_smoke_test_data(self):
        # edge case: two very close points, base_point_2 and point_2,
        # form an angle < epsilon
        base_point = gs.array([1.0, 2.0, 3.0, 4.0, 6.0])
        base_point = base_point / gs.linalg.norm(base_point)
        point = base_point + 1e-4 * gs.array([-1.0, -2.0, 1.0, 1.0, 0.1])
        point = point / gs.linalg.norm(point)
        smoke_data = [
            dict(
                connection_args=(4,),
                point=point,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_smoke_test_data(self):
        base_point = gs.array([1.0, 0.0, 0.0, 0.0])
        tangent_vec = gs.array([0.0, 0.0, gs.pi / 6, 0.0])

        smoke_data = [
            dict(
                connection_args=(4,),
                tangent_vec=tangent_vec,
                base_point=base_point,
            )
        ]

        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=gs.pi / 2.0)

    def exp_and_dist_and_projection_to_tangent_space_test_data(self):
        unnorm_base_point = gs.array([16.0, -2.0, -2.5, 84.0, 3.0])
        base_point = unnorm_base_point / gs.linalg.norm(unnorm_base_point)
        smoke_data = [
            dict(
                dim=4,
                vector=gs.array([9.0, 0.0, -1.0, -2.0, 1.0]),
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)
