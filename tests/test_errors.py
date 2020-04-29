"""Unit tests for errors."""

import geomstats.backend as gs
import geomstats.errors
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.spd_matrices import SPDMatrices


class TestBackends(geomstats.tests.TestCase):
    def test_check_belongs(self):
        euclidean = Euclidean(5)
        point = gs.array([1, 2])

        self.assertRaises(
            RuntimeError,
            lambda: geomstats.errors.check_belongs(
                point, euclidean))

    @staticmethod
    def test_check_belongs_with_tol():
        spd = SPDMatrices(5)
        point = spd.random_uniform()

        geomstats.errors.check_belongs(point, spd, atol=1e-5)

    def test_check_integer(self):
        a = -2

        self.assertRaises(
            ValueError,
            lambda: geomstats.errors.check_integer(a, 'a'))

    def test_check_parameter_accepted_values(self):
        param = 'lefttt'
        accepted_values = ['left', 'right']
        self.assertRaises(
            ValueError,
            lambda: geomstats.errors.check_parameter_accepted_values(
                param, 'left_or_right', accepted_values))
