from .riemannian_metric import RiemannianMetricTestData


class ComplexRiemannianMetricTestData(RiemannianMetricTestData):
    def test_inner_product_is_complex(self):
        return self.generate_random_data()

    def test_dist_is_real(self):
        return self.generate_random_data()

    def test_log_is_complex(self):
        return self.generate_random_data()

    def test_exp_is_complex(self):
        return self.generate_random_data()
