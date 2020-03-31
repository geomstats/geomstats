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

        self.space_matrix = ProductManifold(
            manifolds=[Hypersphere(dimension=2), Hyperbolic(dimension=2)],
            default_point_type='matrix')
        self.space_vector = ProductManifold(
            manifolds=[Hypersphere(dimension=2), Hyperbolic(dimension=5)],
            default_point_type='vector')

    def test_dimension(self):
        expected = 7
        result = self.space.dimension
        self.assertAllClose(result, expected)

    def test_random_and_belongs_matrix(self):
        n_samples = 5
        data = self.space_matrix.random_uniform(n_samples)
        result = self.space_matrix.belongs(data)
        expected = gs.array([[True] * n_samples]).transpose(1, 0)
        self.assertAllClose(result, expected)

    def test_random_and_belongs_vector(self):
        n_samples = 5
        data = self.space_vector.random_uniform(n_samples)
        result = self.space_vector.belongs(data)
        expected = gs.array([[True] * n_samples]).transpose(1, 0)
        self.assertAllClose(result, expected)
