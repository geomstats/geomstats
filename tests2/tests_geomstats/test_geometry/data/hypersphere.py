import random

from geomstats.test.data import TestData

from .base import LevelSetTestData
from .manifold import ManifoldTestData


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


class _HypersphereMixinsTestData:
    pass


class HypersphereExtrinsicTestData(_HypersphereMixinsTestData, LevelSetTestData):
    tolerances = {
        "random_von_mises_fisher_sample_mean": {"atol": 1e-2},
        "random_von_mises_fisher_sample_kappa": {"atol": 1e-1},
        "random_riemannian_normal_frechet_mean": {"atol": 1e-1},
    }
    xfails = ("random_riemannian_normal_frechet_mean",)

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


class HypersphereIntrinsicTestData(_HypersphereMixinsTestData, ManifoldTestData):
    skips = (
        "to_tangent_vec",
        "to_tangent_is_tangent",
    )
