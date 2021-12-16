"""Unit tests for the sub-Riemannian metric class."""

import warnings

import geomstats.backend as gs
import geomstats.tests

class TestSubRiemannianMetric(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=UserWarning)
        self.dim = 3
        self.dist_dim = 2

        new_sub_riemannian_metric = SubRiemannianMetric(dim=3, dist_dim=2)
y
