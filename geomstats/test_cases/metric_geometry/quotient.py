import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test_cases.metric_geometry.point_set import (
    PointSetMetricWithArrayTestCase,
)


class AlignerTestCase(TestCase):
    def test_align(self, point, base_point, expected, atol=gs.atol):
        """Check alignment.

        Parameters
        ----------
        point : array-like, shape=[..., *total_space.shape]
            Point to align.
        base_point : array-like, shape=[..., *total_space.shape]
            Point to align to.
        expected : array-like, shape=[..., *total_space.shape]
            Expected aligned point.
        atol : float
            Absolute tolerance.
        """
        res = self.total_space.aligner.align(point, base_point)
        self.assertAllClose(res, expected, atol=atol)


class QuotientMetricWithArrayTestCase(PointSetMetricWithArrayTestCase):
    """Quotient metric test cases.

    Applies to discrete group actions.
    """

    def test_geodesic_boundary_points(self, initial_point, end_point, atol=gs.atol):
        """Check geodesic goes from the initial point to the aligned end point.

        Parameters
        ----------
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        end_point : array-like, shape=[..., *space.shape]
            End point of the geodesic.
        atol : float
            Absolute tolerance.
        """
        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)

        res = geod_func(gs.array([0.0, 1.0]))
        aligned_end_point = self.total_space.aligner.align(end_point, initial_point)
        expected = gs.stack(
            [initial_point, aligned_end_point], axis=-(self.space.point_ndim + 1)
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_geodesic_bvp_reverse(self, initial_point, end_point, time, atol=gs.atol):
        """Check geodesic is the aligned reverse of the reversed geodesic.

        Parameters
        ----------
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        end_point : array-like, shape=[..., *space.shape]
            End point of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesics.
        atol : float
            Absolute tolerance.
        """
        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)
        geod_func_reverse = self.space.metric.geodesic(
            end_point, end_point=initial_point
        )

        res = geod_func(time)
        res_ = geod_func_reverse(1.0 - time)
        if initial_point.ndim > self.space.point_ndim:
            aligned_res_ = gs.stack(
                [
                    self.total_space.aligner.align(point_res_, initial_point_)
                    for point_res_, initial_point_ in zip(res_, initial_point)
                ]
            )
        else:
            aligned_res_ = self.total_space.aligner.align(res_, initial_point)

        self.assertAllClose(res, aligned_res_, atol=atol)
