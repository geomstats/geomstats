"""Unit tests for errors."""

import pytest

import geomstats.backend as gs
import geomstats.errors
import tests.conftest
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.spd_matrices import SPDMatrices


class TestBackends(tests.conftest.TestCase):
    def test_check_belongs(self):
        euclidean = Euclidean(5)
        point = gs.array([1, 2])

        with pytest.raises(RuntimeError):
            geomstats.errors.check_belongs(point, euclidean)

    @staticmethod
    def test_check_belongs_with_tol():
        spd = SPDMatrices(5)
        point = spd.random_point()

        geomstats.errors.check_belongs(point, spd)

    def test_check_integer(self):
        a = -2

        with pytest.raises(ValueError):
            geomstats.errors.check_integer(a, "a")

    def test_check_parameter_accepted_values(self):
        param = "lefttt"
        accepted_values = ["left", "right"]
        with pytest.raises(ValueError):
            geomstats.errors.check_parameter_accepted_values(
                param, "left_or_right", accepted_values
            )

    def test_check_point_shape(self):
        euclidean = Euclidean(5)
        point = gs.array([1, 2])

        with pytest.raises(geomstats.errors.ShapeError):
            geomstats.errors.check_point_shape(point, euclidean)

        same_shape = geomstats.errors.check_point_shape(
            point, euclidean, suppress_error=True
        )
        self.assertFalse(same_shape)
