"""Unit tests for the sub-Riemannian metric class."""

import warnings

import geomstats.tests
from geomstats.geometry.sub_riemannian_metric import SubRiemannianMetric


class TestSubRiemannianMetric(geomstats.tests.TestCase):
    def setup_method(self):
        warnings.simplefilter("ignore", category=UserWarning)
