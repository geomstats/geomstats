import geomstats.backend as gs
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class SasakiMetricTestCase(RiemannianMetricTestCase):
    def test_geodesic_discrete(self, initial_point, end_point, expected, atol=gs.atol):
        """Check discrete geodesic.

        Parameters
        ----------
        initial_point : array-like, shape=[..., 2, *base_shape]
            Initial point of the geodesic.
        end_point : array-like, shape=[..., 2, *base_shape]
            End point of the geodesic.
        expected : array-like, shape=[..., n_steps + 1, 2, *base_shape]
            Expected points of the discrete geodesic.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.geodesic_discrete(initial_point, end_point)
        self.assertAllClose(res, expected, atol=atol)
