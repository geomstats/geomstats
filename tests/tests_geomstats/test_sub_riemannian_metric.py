"""Unit tests for the sub-Riemannian metric class."""

import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.sub_riemannian_metric import SubRiemannianMetric


class TestSubRiemannianMetric(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=UserWarning)
        self.dim = 3
        self.dist_dim = 2

        new_sub_riemannian_metric = SubRiemannianMetric(dim=self.dim,
                                                        dist_dim=self.dist_dim)
