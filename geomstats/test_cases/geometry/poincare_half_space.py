import geomstats.backend as gs
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class PoincareHalfSpaceMetricTestCase(RiemannianMetricTestCase):
    def test_exp_and_coordinates_tangent(self, tangent_vec, base_point, atol=gs.atol):
        """Check exponential along a vertical tangent vector.

        Geodesics with a vertical initial velocity are vertical lines, so
        the first coordinate is preserved.

        Parameters
        ----------
        tangent_vec : array-like, shape=[2]
            Vertical tangent vector at base point.
        base_point : array-like, shape=[2]
            Point in the half-space.
        atol : float
            Absolute tolerance.
        """
        end_point = self.space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(base_point[0], end_point[0], atol=atol)
