"""Tests for the warped product metric.

Lead author: sdoygb.
"""

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.product_manifold import (
    ProductManifold,
    WarpedProductManifold,
    WarpedProductMetric,
)
from geomstats.test.test_case import assert_allclose

AUTODIFF_PRESENT = gs.has_autodiff()


class TestWarpedProductMetricStructure:
    """Tests of the metric structure, valid on any backend."""

    @staticmethod
    def _euclidean_warped(base_dim=1, fiber_dim=2, warping=None):
        if warping is None:
            warping = lambda r: 1.0 + r**2
        return WarpedProductManifold(
            Euclidean(dim=base_dim), Euclidean(dim=fiber_dim), warping
        )

    def test_inner_product_matches_analytic(self):
        """inner_product equals g_B + f(b)^2 g_F."""
        warp = self._euclidean_warped()
        metric = warp.metric
        base_point = gs.array([1.5, 0.3, -0.7])
        tangent_vec_a = gs.array([0.2, 1.0, -0.5])
        tangent_vec_b = gs.array([-0.4, 0.3, 2.0])
        f = 1.0 + base_point[0] ** 2

        expected = tangent_vec_a[0] * tangent_vec_b[0] + f**2 * gs.dot(
            tangent_vec_a[1:], tangent_vec_b[1:]
        )
        result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        assert_allclose(result, expected)

    def test_inner_product_batched(self):
        """Batched inner products match the per-point results."""
        warp = self._euclidean_warped()
        metric = warp.metric
        base_point = gs.array([1.5, 0.3, -0.7])
        tangent_vec_a = gs.array([0.2, 1.0, -0.5])
        tangent_vec_b = gs.array([-0.4, 0.3, 2.0])

        base_points = gs.stack([base_point, 2.0 * base_point])
        tangent_vecs_a = gs.stack([tangent_vec_a, 2.0 * tangent_vec_a])
        tangent_vecs_b = gs.stack([tangent_vec_b, tangent_vec_b])

        result = metric.inner_product(tangent_vecs_a, tangent_vecs_b, base_points)
        expected = gs.stack(
            [
                metric.inner_product(tangent_vec_a, tangent_vec_b, base_point),
                metric.inner_product(
                    2.0 * tangent_vec_a, tangent_vec_b, 2.0 * base_point
                ),
            ]
        )
        assert_allclose(result, expected)

    def test_squared_norm_matches_inner_product(self):
        """squared_norm is the inner product of a vector with itself."""
        warp = self._euclidean_warped()
        metric = warp.metric
        base_point = gs.array([1.5, 0.3, -0.7])
        vector = gs.array([0.2, 1.0, -0.5])
        result = metric.squared_norm(vector, base_point)
        expected = metric.inner_product(vector, vector, base_point)
        assert_allclose(result, expected)

    def test_metric_matrix_block_structure(self):
        """The metric matrix is block diagonal with the fibre rescaled by f^2."""
        warp = self._euclidean_warped()
        metric = warp.metric
        base_point = gs.array([1.5, 0.3, -0.7])
        f = 1.0 + base_point[0] ** 2
        expected = gs.array(
            [[1.0, 0.0, 0.0], [0.0, f**2, 0.0], [0.0, 0.0, f**2]]
        )
        result = metric.metric_matrix(base_point)
        assert_allclose(result, expected)

    def test_constant_warping_scales_fibre(self):
        """A constant warping f == c rescales the fibre block by c^2."""
        warp = self._euclidean_warped(warping=lambda r: 3.0)
        metric = warp.metric
        base_point = gs.array([1.5, 0.3, -0.7])
        expected = gs.array(
            [[1.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]]
        )
        result = metric.metric_matrix(base_point)
        assert_allclose(result, expected)

    def test_warping_scales_fibre_norm(self):
        """The norm of a fibre vector is rescaled by f (and f^2 in sq. norm)."""
        warp = self._euclidean_warped()
        metric = warp.metric
        fiber_vec = gs.array([0.0, 1.0, 0.0])
        base_1 = gs.array([1.0, 0.0, 0.0])
        base_2 = gs.array([2.0, 0.0, 0.0])
        f1, f2 = 1.0 + 1.0, 1.0 + 4.0
        result = metric.norm(fiber_vec, base_2) / metric.norm(fiber_vec, base_1)
        assert_allclose(result, f2 / f1)

    def test_polar_metric_matrix(self):
        """g = dr^2 + r^2 dy^2 for the warping f(r) = r."""
        warp = WarpedProductManifold(
            Euclidean(dim=1), Euclidean(dim=1), lambda r: r
        )
        metric = warp.metric
        for r, y in [(2.0, 0.5), (1.3, -0.7), (0.7, 1.9)]:
            expected = gs.array([[1.0, 0.0], [0.0, r**2]])
            result = metric.metric_matrix(gs.array([r, y]))
            assert_allclose(result, expected)

    def test_requires_two_factors(self):
        """The warped product metric requires exactly two factors."""
        space = ProductManifold(
            [Euclidean(dim=1), Euclidean(dim=1), Euclidean(dim=1)], equip=False
        )
        with pytest.raises(ValueError):
            WarpedProductMetric(space, lambda r: 1.0 + r**2)

    def test_base_point_is_required(self):
        """The metric matrix requires a base point."""
        warp = self._euclidean_warped()
        with pytest.raises(ValueError):
            warp.metric.metric_matrix()


@pytest.mark.smoke
@pytest.mark.skipif(
    not AUTODIFF_PRESENT, reason="requires an automatic differentiation backend"
)
class TestWarpedProductMetricDifferential:
    """Tests of exp, log and geodesics (require autodiff)."""

    @staticmethod
    def _polar_warped():
        """g = dr^2 + r^2 dy^2: the Euclidean plane in polar-like coordinates."""
        return WarpedProductManifold(
            Euclidean(dim=1), Euclidean(dim=1), lambda r: r
        )

    @staticmethod
    def _sphere_like_warped():
        """g = dtheta^2 + sin^2(theta) dphi^2: the round sphere, local chart."""
        return WarpedProductManifold(
            Euclidean(dim=1), Euclidean(dim=1), lambda theta: gs.sin(theta)
        )

    @staticmethod
    def _polar_to_cartesian(r, y):
        return r * gs.cos(y), r * gs.sin(y)

    @staticmethod
    def _cartesian_to_polar(x, y):
        return gs.sqrt(x**2 + y**2), gs.arctan2(y, x)

    def test_exp_matches_euclidean_polar(self):
        """For g = dr^2 + r^2 dy^2, exp is a straight line in cartesian coords."""
        metric = self._polar_warped().metric
        r0, y0, v_r, v_y = 2.0, 0.6, 0.2, 0.35
        base_point = gs.array([r0, y0])
        tangent_vec = gs.array([v_r, v_y])

        x0, cart_y0 = self._polar_to_cartesian(r0, y0)
        vx = v_r * gs.cos(y0) - r0 * v_y * gs.sin(y0)
        vy = v_r * gs.sin(y0) + r0 * v_y * gs.cos(y0)
        r1, y1 = self._cartesian_to_polar(x0 + vx, cart_y0 + vy)

        result = metric.exp(tangent_vec, base_point)
        expected = gs.array([r1, y1])
        assert_allclose(result, expected, atol=1e-3)

    def test_log_matches_euclidean_polar(self):
        """log inverts the straight-line exp of the polar metric."""
        metric = self._polar_warped().metric
        r0, y0, v_r, v_y = 2.0, 0.6, 0.2, 0.35
        base_point = gs.array([r0, y0])
        tangent_vec = gs.array([v_r, v_y])
        end_point = metric.exp(tangent_vec, base_point)

        result = metric.log(end_point, base_point)
        assert_allclose(result, tangent_vec, atol=1e-2)

    def test_geodesic_matches_euclidean_polar(self):
        """Midpoints of warped geodesics are straight lines in cartesian coords."""
        metric = self._polar_warped().metric
        r0, y0, v_r, v_y = 2.0, 0.6, 0.2, 0.35
        base_point = gs.array([r0, y0])
        tangent_vec = gs.array([v_r, v_y])

        x0, cart_y0 = self._polar_to_cartesian(r0, y0)
        vx = v_r * gs.cos(y0) - r0 * v_y * gs.sin(y0)
        vy = v_r * gs.sin(y0) + r0 * v_y * gs.cos(y0)

        t = 0.4
        point = gs.squeeze(
            metric.geodesic(
                initial_point=base_point, initial_tangent_vec=tangent_vec
            )(t)
        )
        r_t, y_t = self._cartesian_to_polar(x0 + t * vx, cart_y0 + t * vy)
        expected = gs.array([r_t, y_t])
        assert_allclose(point, expected, atol=1e-3)

    def test_exp_log_roundtrip(self):
        """exp(log(q, p), p) recovers q for a non-constant warping."""
        warp = WarpedProductManifold(
            Euclidean(dim=1), Euclidean(dim=2), lambda r: 1.0 + r**2
        )
        metric = warp.metric
        base_point = gs.array([1.5, 0.3, -0.7])
        point = gs.array([1.9, -0.4, 0.8])
        tangent_vec = metric.log(point, base_point)
        result = metric.exp(tangent_vec, base_point)
        assert_allclose(result, point, atol=1e-2)

    def test_geodesic_length_matches_initial_speed(self):
        """The length of a warped geodesic over [0, 1] equals |v|."""
        warp = WarpedProductManifold(
            Euclidean(dim=1), Euclidean(dim=2), lambda r: 1.0 + r**2
        )
        metric = warp.metric
        base_point = gs.array([1.5, 0.3, -0.7])
        tangent_vec = gs.array([0.2, 1.0, -0.5])
        geodesic = metric.geodesic(
            initial_point=base_point, initial_tangent_vec=tangent_vec
        )
        end_point = geodesic(1.0)
        distance = metric.dist(base_point, end_point)
        speed = metric.norm(tangent_vec, base_point)
        assert_allclose(distance, speed, atol=1e-2)

    def test_meridian_geodesic_sphere_like(self):
        """For g = dtheta^2 + sin^2theta dphi^2, a meridian is a geodesic."""
        metric = self._sphere_like_warped().metric
        theta0, phi0, v_theta = 1.0, 0.7, 0.4
        base_point = gs.array([theta0, phi0])
        tangent_vec = gs.array([v_theta, 0.0])
        geodesic = metric.geodesic(
            initial_point=base_point, initial_tangent_vec=tangent_vec
        )
        for t in (0.3, 0.6, 1.0):
            expected = gs.array([theta0 + v_theta * t, phi0])
            assert_allclose(gs.squeeze(geodesic(t)), expected, atol=1e-3)

    def test_equator_geodesic_sphere_like(self):
        """For g = dtheta^2 + sin^2theta dphi^2, the equator is a geodesic."""
        metric = self._sphere_like_warped().metric
        base_point = gs.array([gs.pi / 2, 0.4])
        tangent_vec = gs.array([0.0, 0.5])
        point = gs.squeeze(
            metric.geodesic(
                initial_point=base_point, initial_tangent_vec=tangent_vec
            )(0.7)
        )
        expected = gs.array([gs.pi / 2, 0.4 + 0.5 * 0.7])
        assert_allclose(point, expected, atol=1e-3)

    def test_constant_warping_geodesics_are_linear(self):
        """For constant f, geodesics are straight lines in product coords."""
        warp = WarpedProductManifold(
            Euclidean(dim=1), Euclidean(dim=1), lambda r: 3.0
        )
        metric = warp.metric
        base_point = gs.array([1.0, 0.5])
        tangent_vec = gs.array([0.3, -0.2])
        end_point = metric.exp(tangent_vec, base_point)
        expected = base_point + tangent_vec
        assert_allclose(end_point, expected, atol=1e-3)

    def test_warping_couples_base_and_fibre(self):
        """A pure-fibre velocity curves the base component (unlike a product)."""
        metric = self._polar_warped().metric
        base_point = gs.array([2.0, 0.0])
        tangent_vec = gs.array([0.0, 0.5])
        point = gs.squeeze(
            metric.geodesic(
                initial_point=base_point, initial_tangent_vec=tangent_vec
            )(0.5)
        )

        # a product metric would keep r = 2 constant; the warped one curves it
        assert point[0] > base_point[0]
