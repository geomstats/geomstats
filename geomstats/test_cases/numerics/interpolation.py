import pytest

from geomstats.test.test_case import TestCase


class InterpolatorTestCase(TestCase):
    def test_interpolate(self, t, expected, atol):
        res = self.interpolator(t)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_interpolate_with_given_data(self, atol):
        res = self.interpolator(self.times)
        self.assertAllClose(res, self.data, atol=atol)

    @pytest.mark.random
    def test_interpolate_half_interval(self, atol):
        times = self.times[:-1] + 0.5 * (self.times[1:] - self.times[:-1])

        res = self.interpolator(times)

        point_ndim_slc = (slice(None),) * self.interpolator.point_ndim
        expected = self.data[..., :-1, *point_ndim_slc] + 0.5 * (
            self.data[..., 1:, *point_ndim_slc] - self.data[..., :-1, *point_ndim_slc]
        )
        self.assertAllClose(res, expected, atol=atol)
