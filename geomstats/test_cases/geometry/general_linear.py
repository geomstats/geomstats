import geomstats.backend as gs
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.lie_group import MatrixLieGroupTestCase


class GeneralLinearTestCase(MatrixLieGroupTestCase, VectorSpaceOpenSetTestCase):
    def test_orbit(self, point, base_point, time, expected, atol=gs.atol):
        """Check one-parameter orbit of base point passing through point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Target point.
        base_point : array-like, shape=[..., n, n]
            Base point.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the orbit.
        expected : array-like, shape=[..., n_times, n, n]
            Expected points on the orbit.
        atol : float
            Absolute tolerance.
        """
        path = self.space.orbit(point, base_point)
        res = path(time)
        self.assertAllClose(res, expected, atol=atol)
