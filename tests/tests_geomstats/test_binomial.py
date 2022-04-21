"""Unit tests for the binomial manifold."""

from scipy.stats import binom

import geomstats.backend as gs
import geomstats.tests
from geomstats.information_geometry.binomial import BinomialDistributions
from tests.conftest import Parametrizer
from tests.data.binomial_data import BinomialTestData
from tests.geometry_test_cases import OpenSetTestCase


class TestBinomial(OpenSetTestCase, metaclass=Parametrizer):

    space = BinomialDistributions
    testing_data = BinomialTestData()

    def test_belongs(self, n_draws, point, expected):
        self.assertAllClose(self.space(n_draws).belongs(point), expected)

    def test_random_point(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample(self, n_draws, point, n_samples, expected):
        self.assertAllClose(
            self.space(n_draws).sample(point, n_samples).shape, expected
        )

    def test_squared_dist(self, n_draws, point_a, point_b, expected):
        self.assertAllClose(
            self.space(n_draws).metric.squared_dist(point_a, point_b), expected
        )

    @geomstats.tests.np_and_autograd_only
    def test_point_to_pdf(self, n_draws, point, n_samples):
        point = gs.to_ndarray(point, 1)
        n_points = point.shape[0]
        pmf = self.space(n_draws).point_to_pmf(point)
        samples = gs.to_ndarray(self.space(n_draws).sample(point, n_samples), 1)
        result = gs.squeeze(pmf(samples))
        pmf = []
        for i in range(n_points):
            pmf.append(gs.array([binom.pmf(x, n_draws, point[i]) for x in samples]))
        expected = gs.squeeze(gs.stack(pmf, axis=0))
        self.assertAllClose(result, expected)
