import geomstats.backend as gs
from geomstats.test.test_case import TestCase


class InterpolatorTestCase(TestCase):
    """Interpolator test case.

    The interpolator is built from ``data`` sampled at ``times``.
    """

    def test_interpolate(self, t, expected, atol=gs.atol):
        """Check interpolation.

        Parameters
        ----------
        t : array-like, shape=[n_times,]
            Interpolation times.
        expected : array-like, shape=[..., n_times, *point_shape]
            Expected interpolated points.
        atol : float
            Absolute tolerance.
        """
        res = self.interpolator(t)
        self.assertAllClose(res, expected, atol=atol)

    def test_interpolate_at_nodes(self, atol=gs.atol):
        """Check interpolation recovers the data at the sampled times.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        self.test_interpolate(self.times, self.data, atol=atol)

    def test_interpolate_half_interval(self, atol=gs.atol):
        """Check interpolation at the middle of each interval.

        Linear interpolation gives the midpoint of the data at both ends.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        times = self.times[:-1] + 0.5 * (self.times[1:] - self.times[:-1])

        point_ndim_slc = (slice(None),) * self.interpolator.point_ndim
        expected = self.data[(..., slice(0, -1)) + point_ndim_slc] + 0.5 * (
            self.data[(..., slice(1, None)) + point_ndim_slc]
            - self.data[(..., slice(0, -1)) + point_ndim_slc]
        )
        self.test_interpolate(times, expected, atol=atol)
