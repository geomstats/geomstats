"""Unit tests for the exponential manifold."""

from scipy.stats import expon

import geomstats.backend as gs
import geomstats.tests
from tests.conftest import Parametrizer
from tests.data.exponential_data import ExponentialTestData
from tests.geometry_test_cases import OpenSetTestCase


class TestExponential(OpenSetTestCase, metaclass=Parametrizer):

    testing_data = ExponentialTestData()
    space = testing_data.space

    def test_belongs(self, point, expected):
        self.assertAllClose(self.space().belongs(point), expected)

    def test_random_point(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample(self, point, n_samples, expected):
        self.assertAllClose(self.space().sample(point, n_samples).shape, expected)

    def test_squared_dist(self, point_a, point_b, expected):
        self.assertAllClose(
            self.space().metric.squared_dist(point_a, point_b), expected
        )

    @geomstats.tests.np_and_autograd_only
    def test_point_to_pdf(self, point, n_samples):
        point = gs.to_ndarray(point, 1)
        n_points = point.shape[0]
        pdf = self.space().point_to_pdf(point)
        samples = gs.to_ndarray(self.space().sample(point, n_samples), 1)
        result = gs.squeeze(pdf(samples))
        pdf = []
        for i in range(n_points):
            pdf.append(gs.array([expon.pdf(x, scale=point[i]) for x in samples]))
        expected = gs.squeeze(gs.stack(pdf, axis=0))
        self.assertAllClose(result, expected)
