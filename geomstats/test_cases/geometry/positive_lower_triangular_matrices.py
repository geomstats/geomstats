import geomstats.backend as gs
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class CholeskyMetricTestCase(RiemannianMetricTestCase):
    def test_diag_inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol=gs.atol
    ):
        """Check inner product of the diagonal parts.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., n, n]
            Positive lower triangular matrix.
        expected : array-like, shape=[...]
            Expected inner product.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.diag_inner_product(
            tangent_vec_a,
            tangent_vec_b,
            base_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_strictly_lower_inner_product(
        self, tangent_vec_a, tangent_vec_b, expected, atol=gs.atol
    ):
        """Check inner product of the strictly lower triangular parts.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector.
        expected : array-like, shape=[...]
            Expected inner product.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.strictly_lower_inner_product(
            tangent_vec_a,
            tangent_vec_b,
        )
        self.assertAllClose(res, expected, atol=atol)
