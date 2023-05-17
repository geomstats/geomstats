from geomstats.test.data import TestData


class FiberBundleTestData(TestData):
    def riemannian_submersion_vec_test_data(self):
        return self.generate_vec_data()

    def riemannian_submersion_belongs_to_base_test_data(self):
        return self.generate_random_data()

    def lift_vec_test_data(self):
        return self.generate_vec_data()

    def lift_belongs_to_total_space_test_data(self):
        return self.generate_random_data()

    def riemannian_submersion_after_lift_test_data(self):
        return self.generate_random_data()

    def tangent_riemannian_submersion_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_riemannian_submersion_is_tangent_test_data(self):
        return self.generate_random_data()

    def align_vec_test_data(self):
        return self.generate_vec_data()

    def log_after_align_is_horizontal_test_data(self):
        return self.generate_random_data()

    def horizontal_projection_vec_test_data(self):
        return self.generate_vec_data()

    def horizontal_projection_is_horizontal_test_data(self):
        return self.generate_random_data()

    def vertical_projection_vec_test_data(self):
        return self.generate_vec_data()

    def vertical_projection_is_vertical_test_data(self):
        return self.generate_random_data()

    def tangent_riemannian_submersion_after_vertical_projection_test_data(self):
        return self.generate_random_data()

    def is_horizontal_vec_test_data(self):
        return self.generate_vec_data()

    def is_vertical_vec_test_data(self):
        return self.generate_vec_data()

    def horizontal_lift_vec_test_data(self):
        return self.generate_vec_data()

    def horizontal_lift_is_horizontal_test_data(self):
        return self.generate_random_data()

    def tangent_riemannian_submersion_after_horizontal_lift_test_data(self):
        return self.generate_random_data()

    def integrability_tensor_vec_test_data(self):
        return self.generate_vec_data()

    def integrability_tensor_derivative_vec_test_data(self):
        return self.generate_vec_data()


class GeneralLinearBuresWassersteinBundleTestData(FiberBundleTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
    xfails = (
        "align_vec",
        "exp_after_log",
        "log_after_exp",
        "log_after_align_is_horizontal",
        "horizontal_lift_is_horizontal",
    )

    tolerances = {
        "align_vec": {"atol": 1e-1},
        "horizontal_lift_is_horizontal": {"atol": 1e-1},
        "log_after_exp": {"atol": 1e-1},
        "log_after_align_is_horizontal": {"atol": 1e-1},
    }
