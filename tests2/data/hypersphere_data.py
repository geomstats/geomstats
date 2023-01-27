import geomstats.backend as gs
from geomstats.test.data import TestData
from tests2.data.base_data import LevelSetTestData, ManifoldTestData


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
    }

    def _get_random_kappa(self, size=1):
        sample = gs.random.uniform(low=1.0, high=10000.0, size=(size,))
        if size == 1:
            return sample[0]
        return sample

    def random_von_mises_fisher_belongs_test_data(self):
        data = []
        for random_mu in [True, False]:
            data.extend(
                [
                    dict(
                        n_points=n_points,
                        random_mu=random_mu,
                        kappa=self._get_random_kappa(),
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
                        kappa=self._get_random_kappa(),
                    )
                    for n_points in self.N_SHAPE_POINTS
                ]
            )

        return self.generate_tests(data)

    def random_von_mises_fisher_sample_mean_test_data(self):
        n_samples = 10000
        data = []
        for kappa in self._get_random_kappa(size=2):
            data.append(dict(n_samples=n_samples, kappa=kappa))

        data.append(
            dict(
                n_samples=n_samples,
                kappa=self._get_random_kappa(),
                random_mu=False,
            )
        )

        return self.generate_tests(data)

    def random_von_mises_fisher_sample_kappa_test_data(self):
        data = [dict(n_samples=50000)]
        return self.generate_tests(data)


class HypersphereIntrinsicTestData(_HypersphereMixinsTestData, ManifoldTestData):
    skips = (
        "to_tangent_vec",
        "to_tangent_is_tangent",
    )
