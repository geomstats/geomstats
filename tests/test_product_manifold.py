"""
Unit tests for ProductManifold.
"""


import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.product_manifold import ProductManifold


class TestProductManifoldMethods(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.sphere = Hypersphere(dimension=2)
        self.hyperbolic = Hyperbolic(dimension=5)

        self.space = ProductManifold(
            manifolds=[self.sphere, self.hyperbolic])

    def test_dimension(self):
        expected = 7
        result = self.space.dimension
        self.assertAllClose(result, expected)
