import geomstats.backend as gs
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


class DirichletDistributionsTestCase(
    InformationManifoldMixinTestCase, VectorSpaceOpenSetTestCase
):
    def _check_sample_belongs_to_support(self, sample, atol):
        self.assertTrue(gs.all(sample >= 0.0))
        self.assertTrue(gs.all(gs.abs(gs.sum(sample, axis=-1) - 1.0) < atol))


class DirichletMetricTestCase(RiemannianMetricTestCase):
    def test_jacobian_christoffels(self, base_point, expected, atol=gs.atol):
        """Check Jacobian of the Christoffel symbols.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        expected : array-like, shape=[..., dim, dim, dim, dim]
            Expected Jacobian.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.jacobian_christoffels(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_sectional_curvature_is_negative(
        self, base_point, tangent_vec_a, tangent_vec_b
    ):
        """Check sectional curvature is non-positive.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        tangent_vec_a : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        """
        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertTrue(gs.all(res <= 0))

    def test_exp_diagonal_is_totally_geodesic(
        self, param, tangent_param, atol=gs.atol
    ):
        """Check the diagonal x1 = ... = xn is totally geodesic.

        Parameters
        ----------
        param : float
            Value of the coordinates of the base point.
        tangent_param : float
            Value of the coordinates of the tangent vector.
        atol : float
            Absolute tolerance.
        """
        base_point = param * gs.ones(self.space.shape)
        tangent_vec = tangent_param * gs.ones(self.space.shape)

        end_point = self.space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(
            end_point - end_point[0], gs.zeros(self.space.shape), atol=atol
        )
