from .riemannian_metric import RiemannianMetricTestData


class FlatRiemannianMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def inner_product_derivative_matrix_is_zeros_test_data(self):
        return self.generate_random_data()

    def christoffels_are_zeros_test_data(self):
        return self.generate_random_data()
