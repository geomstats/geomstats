from geomstats.test.data import TestData


class DiffeoTestData(TestData):
    trials = 1

    def diffeomorphism_vec_test_data(self):
        return self.generate_vec_data()

    def diffeomorphism_belongs_test_data(self):
        return self.generate_random_data()

    def inverse_diffeomorphism_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_diffeomorphism_belongs_test_data(self):
        return self.generate_random_data()

    def inverse_diffeomorphism_after_diffeomorphism_test_data(self):
        return self.generate_random_data()

    def diffeomorphism_after_inverse_diffeomorphism_test_data(self):
        return self.generate_random_data()

    def tangent_diffeomorphism_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_diffeomorphism_is_tangent_test_data(self):
        return self.generate_random_data()

    def tangent_diffeomorphism_with_image_point_test_data(self):
        return self.generate_random_data()

    def inverse_tangent_diffeomorphism_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_tangent_diffeomorphism_is_tangent_test_data(self):
        return self.generate_random_data()

    def inverse_tangent_diffeomorphism_with_base_point_test_data(self):
        return self.generate_random_data()

    def inverse_tangent_diffeomorphism_after_tangent_diffeomorphism_test_data(self):
        return self.generate_random_data()

    def tangent_diffeomorphism_after_inverse_tangent_diffeomorphism_test_data(self):
        return self.generate_random_data()


class AutodiffDiffeoTestData(DiffeoTestData):
    fail_for_autodiff_exceptions = False

    def jacobian_diffeomorphism_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_jacobian_diffeomorphism_vec_test_data(self):
        return self.generate_vec_data()


class DiffeoComparisonTestData(TestData):
    fail_for_autodiff_exceptions = False

    def diffeomorphism_test_data(self):
        return self.generate_random_data()

    def inverse_diffeomorphism_test_data(self):
        return self.generate_random_data()

    def tangent_diffeomorphism_test_data(self):
        return self.generate_random_data()

    def inverse_tangent_diffeomorphism_test_data(self):
        return self.generate_random_data()
