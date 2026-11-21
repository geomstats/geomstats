"""Tests for the Laplace-Beltrami operator.

Lead author: sdoygb.
"""

import pytest

import geomstats.backend as gs
from geomstats.exceptions import AutodiffNotImplementedError
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.product_manifold import WarpedProductManifold
from geomstats.test.test_case import assert_allclose

AUTODIFF_PRESENT = gs.has_autodiff()


@pytest.mark.smoke
@pytest.mark.skipif(
    not AUTODIFF_PRESENT, reason="requires an automatic differentiation backend"
)
class TestLaplacian:
    """Tests of the scalar Laplacian against closed forms."""

    @staticmethod
    def _sphere_like_warped():
        """g = dtheta^2 + sin^2(theta) dphi^2: the round sphere, local chart."""
        return WarpedProductManifold(
            Euclidean(dim=1), Euclidean(dim=1), lambda theta: gs.sin(theta)
        )

    @staticmethod
    def _polar_warped():
        """g = dr^2 + r^2 dy^2: the Euclidean plane in polar-like coordinates."""
        return WarpedProductManifold(
            Euclidean(dim=1), Euclidean(dim=1), lambda r: r
        )

    def test_euclidean_quadratic(self):
        """On Euclidean space, Delta(x^2 + y^2) = 4."""
        metric = Euclidean(dim=2).metric
        function = lambda point: point[0] ** 2 + point[1] ** 2
        for point in [(1.0, 2.0), (-0.5, 3.0), (2.0, -1.0)]:
            result = metric.laplacian(function, gs.array(point))
            assert_allclose(result, 4.0, atol=1e-10)

    def test_sphere_spherical_harmonic(self):
        """On the round sphere, Delta(cos theta) = -2 cos theta (ell = 1)."""
        metric = self._sphere_like_warped().metric
        function = lambda point: gs.cos(point[0])
        for point in [(1.0, 0.5), (0.6, 2.0), (1.4, -0.8)]:
            result = metric.laplacian(function, gs.array(point))
            expected = -2.0 * gs.cos(point[0])
            assert_allclose(result, expected, atol=1e-8)

    def test_warped_polar_decomposition(self):
        """Warped-product Laplacian decomposition on g = dr^2 + r^2 dy^2.

        For M = B x_f F the Laplacian splits as
        Delta_M = Delta_B + f^{-2} Delta_F + m <grad_B ln f, grad_B .>,
        with m = dim F = 1. For f_M(r, y) = r^2 + cos(y) this gives
        Delta f = 4 - cos(y) / r^2.
        """
        metric = self._polar_warped().metric
        function = lambda point: point[0] ** 2 + gs.cos(point[1])
        for point in [(2.0, 0.5), (1.5, 1.2), (3.0, -0.7)]:
            result = metric.laplacian(function, gs.array(point))
            expected = 4.0 - gs.cos(point[1]) / point[0] ** 2
            assert_allclose(result, expected, atol=1e-8)

    def test_linearity(self):
        """Delta(f + g) = Delta f + Delta g."""
        metric = Euclidean(dim=2).metric
        f = lambda point: point[0] ** 2
        g = lambda point: gs.sin(point[1])
        base_point = gs.array([1.2, 0.9])
        result = metric.laplacian(lambda p: f(p) + g(p), base_point)
        expected = metric.laplacian(f, base_point) + metric.laplacian(g, base_point)
        assert_allclose(result, expected, atol=1e-10)

    def test_constant_vanishes(self):
        """Delta of a constant function vanishes."""
        metric = Euclidean(dim=2).metric
        base_point = gs.array([1.2, 0.9])
        result = metric.laplacian(
            lambda point: 3.0 + 0.0 * point[0], base_point
        )
        assert_allclose(result, 0.0, atol=1e-10)

    def test_batched(self):
        """Batched base points give batched Laplacian values."""
        metric = Euclidean(dim=2).metric
        function = lambda point: point[0] ** 2 + point[1] ** 2
        base_points = gs.array([[1.0, 2.0], [0.5, -1.0]])
        result = metric.laplacian(function, base_points)
        assert_allclose(result, gs.array([4.0, 4.0]), atol=1e-10)


@pytest.mark.smoke
@pytest.mark.skipif(AUTODIFF_PRESENT, reason="requires a non-autodiff backend")
class TestLaplacianWithoutAutodiff:
    """Without autodiff, the Laplacian raises a clear error."""

    def test_requires_autodiff(self):
        metric = Euclidean(dim=2).metric
        with pytest.raises(AutodiffNotImplementedError):
            metric.laplacian(lambda p: p[0] ** 2, gs.array([1.0, 2.0]))
