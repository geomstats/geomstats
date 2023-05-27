from .riemannian_metric import RiemannianMetricTestData


class PullbackDiffeoMetricTestData(RiemannianMetricTestData):
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

    def jacobian_diffeomorphism_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_diffeomorphism_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_diffeomorphism_is_tangent_test_data(self):
        return self.generate_random_data()

    def inverse_jacobian_diffeomorphism_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_tangent_diffeomorphism_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_tangent_diffeomorphism_is_tangent_test_data(self):
        return self.generate_random_data()

    def inverse_tangent_diffeomorphism_after_tangent_diffeomorphism_test_data(self):
        return self.generate_random_data()

    def tangent_diffeomorphism_after_inverse_tangent_diffeomorphism_test_data(self):
        return self.generate_random_data()
