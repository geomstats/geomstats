"""Unit tests for the MultivariateDiagonalNormalDistributions manifold."""

import geomstats.backend as gs
import tests.conftest
from tests.conftest import Parametrizer, np_backend, pytorch_backend, tf_backend
from tests.data.multivariate_normal import\
        MultivariateDiagonalNormalDistributionsTestData
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestMultivariateDiagonalNormalDistributions(OpenSetTestCase, metaclass=Parametrizer):
    testing_data = MultivariateDiagonalNormalDistributionsTestData()

    def test_belongs(self, n, point, expected):
        self.assertAllClose(self.Space(n).belongs(point), expected)

    def test_random_point_shape(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample(self, n, point, n_samples, expected):
        self.assertAllClose(
            self.Space(n).sample(point, n_samples).shape, expected)

    # @tests.conftest.np_and_autograd_only
    # def test_point_to_pdf(self, n, point, n_samples):
    #     point = gs.to_ndarray(point, 2)
    #     n_points = point.shape[0]
    #     pdf = self.Space(n).point_to_pdf(point)
    #     alpha = gs.ones(n)
    #     samples = self.Space(n).sample(alpha, n_samples)
    #     result = pdf(samples)
    #     pdf = []
    #     for i in range(n_points):
    #         pdf.append(gs.array([dirichlet.pdf(x, point[i, :]) for x in samples]))
    #     expected = gs.squeeze(gs.stack(pdf, axis=0))
    #     self.assertAllClose(result, expected)
