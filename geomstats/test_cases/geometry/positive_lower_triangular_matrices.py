from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class CholeskyMetricTestCase(RiemannianMetricTestCase):
    def test_diag_inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.diag_inner_product(
            tangent_vec_a,
            tangent_vec_b,
            base_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_strictly_lower_inner_product(
        self, tangent_vec_a, tangent_vec_b, expected, atol
    ):
        res = self.space.metric.strictly_lower_inner_product(
            tangent_vec_a,
            tangent_vec_b,
        )
        self.assertAllClose(res, expected, atol=atol)
