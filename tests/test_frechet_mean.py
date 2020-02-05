"""Unit tests for Frechet mean."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import FrechetMean


class TestFrechetMean(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.so3 = SpecialOrthogonal(n=3)
        self.n_samples = 10

        self.X = self.so3.random_uniform(n_samples=self.n_samples)
        self.metric = self.so3.bi_invariant_metric
        self.n_components = 2

    @geomstats.tests.np_only
    def test_frechet_mean(self):
        X = self.X
        mean = FrechetMean(self.metric)

        mean.fit(X)
        expected =
        self.assertEquals(mean.mean_, expected)
