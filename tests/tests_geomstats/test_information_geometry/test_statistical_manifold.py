import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.information_geometry.statistical_metric import StatisticalManifold
from geomstats.test.test_case import TestCase, autograd_and_torch_only


@autograd_and_torch_only
@pytest.mark.slow
class TestStatisticalMetric(TestCase):
    """"""
# class TestBregmanDivergence(TestCase):
#     """Verify Bregman divergence on Euclidean space for a simple potential function."""

#     def setup_method(self):
#         self.space = Euclidean(dim=2)
#         self.bregman_divergence = BregmanDivergence(
#             self.space, potential_function=lambda point: gs.sum(point**2)
#         )

#     def test_bregman_divergence(self):
#         base_point = gs.array([0.0, 0.0])
#         point = gs.array([1.0, 1.0])
#         result = self.bregman_divergence.bregman_divergence(base_point, point)
#         expected = 2.0
#         self.assertAllClose(result, expected)