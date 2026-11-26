import geomstats.backend as gs
from geomstats.test.test_case import TestCase


class PointToGeodesicAlignerTestCase(TestCase):
    def test_align_with_endpoints(
        self, initial_point, end_point, point, expected, atol=gs.atol
    ):
        """Check alignment of a point to a geodesic.

        Parameters
        ----------
        initial_point : array-like, shape=[..., n_nodes, n_nodes]
            Initial point of the geodesic.
        end_point : array-like, shape=[..., n_nodes, n_nodes]
            End point of the geodesic.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Point to align.
        expected : array-like, shape=[..., n_nodes, n_nodes]
            Expected aligned point.
        atol : float
            Absolute tolerance.
        """
        geodesic = self.total_space.quotient.metric.geodesic(initial_point, end_point)
        res = self.total_space.aligner.align_point_to_geodesic(geodesic, point)
        self.assertAllClose(res, expected, atol=atol)

    def test_dist_with_endpoints(
        self, initial_point, end_point, point, expected, atol=gs.atol
    ):
        """Check distance of a point to a geodesic.

        Parameters
        ----------
        initial_point : array-like, shape=[..., n_nodes, n_nodes]
            Initial point of the geodesic.
        end_point : array-like, shape=[..., n_nodes, n_nodes]
            End point of the geodesic.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Point.
        expected : array-like, shape=[...]
            Expected distance.
        atol : float
            Absolute tolerance.
        """
        geodesic = self.total_space.quotient.metric.geodesic(initial_point, end_point)
        res = self.total_space.aligner.point_to_geodesic_aligner.dist(geodesic, point)
        self.assertAllClose(res, expected, atol=atol)

    def test_dist_along_geodesic_is_zero(self, initial_point, end_point, atol=gs.atol):
        """Check a point of the geodesic is at distance zero from it.

        Parameters
        ----------
        initial_point : array-like, shape=[..., n_nodes, n_nodes]
            Initial point of the geodesic.
        end_point : array-like, shape=[..., n_nodes, n_nodes]
            End point of the geodesic.
        atol : float
            Absolute tolerance.

        Notes
        -----
        The point is taken at a random time along the geodesic.
        """
        geodesic = self.total_space.quotient.metric.geodesic(initial_point, end_point)
        point = gs.squeeze(geodesic(gs.random.rand(1)), axis=-3)

        batch_shape = initial_point.shape[: -self.total_space.point_ndim]
        self.test_dist_with_endpoints(
            initial_point, end_point, point, gs.zeros(batch_shape), atol
        )
