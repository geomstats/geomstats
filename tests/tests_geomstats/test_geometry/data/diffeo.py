from polpo.testing.geometric import GeometricMapCaseData

from geomstats.test.data import TestData


class DiffeoTestData(GeometricMapCaseData):
    def diffeomorphism_belongs_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def inverse_belongs_test_data(self):
        return self.generate_random_data(
            arg_names="image_point",
            data_space="image",
        )

    def inverse_after_diffeomorphism_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def diffeomorphism_after_inverse_test_data(self):
        return self.generate_random_data(
            arg_names="image_point",
            data_space="image",
        )

    def tangent_is_tangent_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def tangent_with_image_point_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def inverse_tangent_is_tangent_test_data(self):
        return self.generate_random_data(
            arg_names=("image_point", "image_tangent_vec"),
            data_space="image",
        )

    def inverse_tangent_with_base_point_test_data(self):
        return self.generate_random_data(
            arg_names=("image_point", "image_tangent_vec"),
            data_space="image",
        )

    def inverse_tangent_after_tangent_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def tangent_after_inverse_tangent_test_data(self):
        return self.generate_random_data(
            arg_names=("image_point", "image_tangent_vec"),
            data_space="image",
        )

    def diffeomorphism_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="base_point",
            op_name="__call__",
        )

    def inverse_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="image_point",
            data_space="image",
        )

    def tangent_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def inverse_tangent_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=(
                "image_point",
                "image_tangent_vec",
            ),
            data_space="image",
        )


class AutodiffDiffeoTestData(DiffeoTestData):
    fail_for_autodiff_exceptions = False

    def jacobian_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_jacobian_vec_test_data(self):
        return self.generate_vec_data()


class DiffeoComparisonTestData(TestData):
    fail_for_autodiff_exceptions = False

    def diffeomorphism_test_data(self):
        return self.generate_random_data()

    def inverse_test_data(self):
        return self.generate_random_data()

    def tangent_test_data(self):
        return self.generate_random_data()

    def inverse_tangent_test_data(self):
        return self.generate_random_data()
