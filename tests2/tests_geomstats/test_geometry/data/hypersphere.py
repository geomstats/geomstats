import random

import pytest

import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import LevelSetTestData
from .manifold import ManifoldTestData
from .riemannian_metric import RiemannianMetricTestData


class HypersphereCoordsTransformTestData(TestData):
    def intrinsic_to_extrinsic_coords_vec_test_data(self):
        return self.generate_vec_data()

    def intrinsic_to_extrinsic_coords_belongs_test_data(self):
        return self.generate_random_data()

    def extrinsic_to_intrinsic_coords_vec_test_data(self):
        return self.generate_vec_data()

    def extrinsic_to_intrinsic_coords_belongs_test_data(self):
        return self.generate_random_data()

    def intrinsic_to_extrinsic_coords_after_extrinsic_to_intrinsic_test_data(self):
        return self.generate_random_data()

    def extrinsic_to_intrinsic_coords_after_intrinsic_to_extrinsic_coords_test_data(
        self,
    ):
        return self.generate_random_data()

    def tangent_spherical_to_extrinsic_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_spherical_to_extrinsic_is_tangent_test_data(self):
        return self.generate_random_data()

    def tangent_extrinsic_to_spherical_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_extrinsic_to_spherical_is_tangent_test_data(self):
        return self.generate_random_data()

    def tangent_extrinsic_to_spherical_after_tangent_spherical_to_extrinsic_test_data(
        self,
    ):
        return self.generate_random_data()

    def tangent_spherical_to_extrinsic_after_tangent_extrinsic_to_spherical_test_data(
        self,
    ):
        return self.generate_random_data()


class HypersphereExtrinsicTestData(LevelSetTestData):
    trials = 2
    tolerances = {
        "random_von_mises_fisher_sample_mean": {"atol": 1e-2},
        "random_von_mises_fisher_sample_kappa": {"atol": 1e-1},
        "random_riemannian_normal_frechet_mean": {"atol": 1e-1},
    }

    def replace_values_test_data(self):
        return self.generate_tests([dict(n_points=random.randint(2, 10))])

    def random_von_mises_fisher_sample_mean_test_data(self):
        n_samples = 1000
        data = []
        for _ in range(2):
            # to test different kappa
            data.append(dict(n_samples=n_samples))

        data.append(
            dict(
                n_samples=n_samples,
                random_mu=False,
            )
        )

        return self.generate_tests(data)

    def random_von_mises_fisher_sample_kappa_test_data(self):
        data = [dict(n_samples=5000)]
        return self.generate_tests(data)

    def random_von_mises_fisher_belongs_test_data(self):
        data = []
        for random_mu in [True, False]:
            data.extend(
                [
                    dict(
                        n_points=n_points,
                        random_mu=random_mu,
                    )
                    for n_points in self.N_RANDOM_POINTS
                ]
            )

        return self.generate_tests(data)

    def random_von_mises_fisher_shape_test_data(self):
        data = []
        for random_mu in [True, False]:
            data.extend(
                [
                    dict(
                        n_points=n_points,
                        random_mu=random_mu,
                    )
                    for n_points in self.N_SHAPE_POINTS
                ]
            )

        return self.generate_tests(data)

    def random_riemannian_normal_belongs_test_data(self):
        data = []
        for random_mean, precision_type in zip(
            [True, False, False],
            [None, float, "array"],
        ):
            for n_samples in self.N_RANDOM_POINTS:
                data.append(
                    dict(
                        n_samples=n_samples,
                        random_mean=random_mean,
                        precision_type=precision_type,
                    )
                )

        return self.generate_tests(data)

    def random_riemannian_normal_shape_test_data(self):
        return self.random_riemannian_normal_belongs_test_data()

    def random_riemannian_normal_frechet_mean_test_data(self):
        data = []
        n_samples = 5000
        for random_mean in [True, False]:
            data.append(dict(n_samples=n_samples, random_mean=random_mean))

        return self.generate_tests(data)


class HypersphereIntrinsicTestData(ManifoldTestData):
    skips = (
        "to_tangent_vec",
        "to_tangent_is_tangent",
        "regularize_belongs",
    )


class HypersphereExtrinsicMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    trials = 3

    tolerances = {"dist_point_to_itself_is_zero": {"atol": 1e-6}}

    def sectional_curvature_is_one_test_data(self):
        return self.generate_random_data()


class Hypersphere2IntrinsicMetricTestData(TestData):
    fail_for_autodiff_exceptions = False

    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([0.3, 0.4]),
                tangent_vec_b=gs.array([0.1, -0.5]),
                base_point=gs.array([gs.pi / 3.0, gs.pi / 5.0]),
                expected=gs.array(-0.12),
            )
        ]
        return self.generate_tests(
            data,
            marks=(
                pytest.mark.smoke,
                pytest.mark.skip,
            ),
        )

    def christoffels_vec_test_data(self):
        return self.generate_vec_data()

    def ricci_tensor_test_data(self):
        theta = gs.pi / 3
        base_point = gs.array([theta, gs.pi / 7])
        expected = gs.array([[1.0, 0.0], [0.0, gs.sin(theta) ** 2]])

        base_points = gs.array(
            [[gs.pi / 3, gs.pi / 7], [gs.pi / 4, gs.pi / 8], [gs.pi / 5, gs.pi / 9]]
        )
        expected_ = gs.array(
            [
                [[1.0, 0.0], [0.0, gs.sin(gs.pi / 3) ** 2]],
                [[1.0, 0.0], [0.0, gs.sin(gs.pi / 4) ** 2]],
                [[1.0, 0.0], [0.0, gs.sin(gs.pi / 5) ** 2]],
            ]
        )
        data = [
            dict(base_point=base_point, expected=expected),
            dict(base_point=base_points, expected=expected_),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def riemann_tensor_spherical_coords_test_data(self):
        data = [dict(base_point=gs.array([gs.pi / 2, gs.pi / 6]))]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([gs.pi / 2.0, 0.0]),
                base_point=gs.array([gs.pi / 10.0, gs.pi / 9.0]),
                expected=gs.array([gs.pi / 10.0 + gs.pi / 2.0, gs.pi / 9.0]),
            )
        ]
        return self.generate_tests(
            data,
            marks=(
                pytest.mark.smoke,
                pytest.mark.skip,
            ),
        )


class Hypersphere2ExtrinsicMetricTestData(TestData):
    def hamiltonian_test_data(self):
        data = [
            dict(
                state=(gs.array([0.0, 0.0, 1.0]), gs.array([1.0, 2.0, 1.0])),
                expected=gs.array(3.0),
            )
        ]
        return self.generate_tests(data)


class Hypersphere4ExtrinsicMetricTestData(TestData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                tangent_vec_b=gs.array([0.0, 1.0, 0.0, 0.0, 0.0]),
                base_point=gs.array([0.0, 0.0, 0.0, 0.0, 1.0]),
                expected=gs.array(0.0),
            ),
        ]
        return self.generate_tests(data)

    def dist_test_data(self):
        point_a = gs.array([10.0, -2.0, -0.5, 0.0, 0.0])
        point_a = point_a / gs.linalg.norm(point_a)
        point_b = gs.array([2.0, 10, 0.0, 0.0, 0.0])
        point_b = point_b / gs.linalg.norm(point_b)
        data = [
            dict(
                point_a=point_a,
                point_b=point_b,
                expected=gs.array(gs.pi / 2),
            ),
            dict(
                point_a=gs.array(
                    [
                        1.0 / gs.sqrt(129.0) * gs.array([10.0, -2.0, -5.0, 0.0, 0.0]),
                    ]
                ),
                point_b=gs.array(
                    [1.0 / gs.sqrt(435.0) * gs.array([1.0, -20.0, -5.0, 0.0, 3.0])]
                ),
                expected=gs.array([1.24864502]),
            ),
        ]
        return self.generate_tests(data)

    def exp_and_dist_and_projection_to_tangent_space_test_data(self):
        unnorm_base_point = gs.array([16.0, -2.0, -2.5, 84.0, 3.0])
        base_point = unnorm_base_point / gs.linalg.norm(unnorm_base_point)
        data = [
            dict(
                vector=gs.array([9.0, 0.0, -1.0, -2.0, 1.0]),
                base_point=base_point,
            )
        ]
        return self.generate_tests(data)
