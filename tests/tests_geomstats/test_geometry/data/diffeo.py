from geomstats.test.data import TestData


class DiffeoTestData(TestData):
    def diffeomorphism_vec_test_data(self):
        return self.generate_vec_data()

    def diffeomorphism_belongs_test_data(self):
        return self.generate_random_data()

    def inverse_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_belongs_test_data(self):
        return self.generate_random_data()

    def inverse_after_diffeomorphism_test_data(self):
        return self.generate_random_data()

    def diffeomorphism_after_inverse_test_data(self):
        return self.generate_random_data()

    def tangent_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_is_tangent_test_data(self):
        return self.generate_random_data()

    def tangent_with_image_point_test_data(self):
        return self.generate_random_data()

    def inverse_tangent_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_tangent_is_tangent_test_data(self):
        return self.generate_random_data()

    def inverse_tangent_with_base_point_test_data(self):
        return self.generate_random_data()

    def inverse_tangent_after_tangent_test_data(self):
        return self.generate_random_data()

    def tangent_after_inverse_tangent_test_data(self):
        return self.generate_random_data()


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
