import geomstats.backend as gs
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class PoincareBallTestCase(VectorSpaceOpenSetTestCase):
    def test_projection_norm_less_than_1(self, point, atol=gs.atol):
        """Check projected point lies in the open unit ball.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to project.
        atol : float
            Absolute tolerance.
        """
        projected_point = self.space.projection(point)
        result = gs.sum(projected_point * projected_point, axis=-1) < 1.0 + atol
        self.assertTrue(result)


class PoincareBallMetricTestCase(RiemannianMetricTestCase):
    def test_mobius_add(
        self, point_a, point_b, expected, atol=gs.atol, project_first=True
    ):
        """Check Mobius addition.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point.
        point_b : array-like, shape=[..., dim]
            Point.
        expected : array-like, shape=[..., dim]
            Expected Mobius sum.
        atol : float
            Absolute tolerance.
        project_first : bool
            Whether to project points before adding them.
        """
        res = self.space.metric.mobius_add(
            point_a, point_b, project_first=project_first
        )
        self.assertAllClose(res, expected, atol=atol)
