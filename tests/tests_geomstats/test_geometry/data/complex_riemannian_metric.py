from .riemannian_metric import RiemannianMetricTestData


class ComplexRiemannianMetricTestData(RiemannianMetricTestData):
    def inner_product_is_complex_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )

    def dist_is_real_test_data(self):
        return self.generate_random_data(arg_names=("point_a", "point_b"))

    def log_is_complex_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"))

    def exp_is_complex_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))
