import geomstats.backend as gs
from geomstats.test.assertions import assert_is_tangent
from geomstats.test_cases.geometry.complex_riemannian_metric import (
    ComplexRiemannianMetricTestCase,
)


class SiegelMetricTestCase(ComplexRiemannianMetricTestCase):
    def test_tangent_vec_from_base_point_to_zero(
        self, tangent_vec, base_point, expected, atol=gs.atol
    ):
        """Check transport of a tangent vector from base point to zero.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.
        expected : array-like, shape=[..., n, n]
            Expected tangent vector at zero.
        atol : float
            Absolute tolerance.
        """
        tangent_vec_at_zero = self.space.metric.tangent_vec_from_base_point_to_zero(
            tangent_vec, base_point
        )
        self.assertAllClose(tangent_vec_at_zero, expected, atol=atol)

    def test_sectional_curvature_at_zero(
        self, tangent_vec_a, tangent_vec_b, expected, atol=gs.atol
    ):
        """Check sectional curvature at zero.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at zero.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at zero.
        expected : array-like, shape=[...,]
            Expected sectional curvature.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.sectional_curvature_at_zero(
            tangent_vec_a, tangent_vec_b
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_tangent_vec_from_base_point_to_zero_is_tangent(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        """Check transport of a tangent vector to zero is tangent at zero.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        tangent_vec_at_zero = self.space.metric.tangent_vec_from_base_point_to_zero(
            tangent_vec, base_point
        )
        zero = gs.zeros(self.space.shape)
        assert_is_tangent(self.space, tangent_vec_at_zero, zero, atol=atol)
