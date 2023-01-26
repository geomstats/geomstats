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
    pass


class HypersphereIntrinsicTestData(_HypersphereMixinsTestData, ManifoldTestData):
    skips = (
        "to_tangent_vec",
        "to_tangent_is_tangent",
    )
