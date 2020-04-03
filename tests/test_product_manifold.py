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
        result = self.space_vector.dimension
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_random_and_belongs_matrix(self):
        n_samples = 5
        data = self.space_matrix.random_uniform(n_samples)
        result = self.space_matrix.belongs(data)
        expected = gs.array([[True] * n_samples]).transpose(1, 0)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_random_and_belongs_vector(self):
        n_samples = 5
        data = self.space_vector.random_uniform(n_samples)
        result = self.space_vector.belongs(data)
        expected = gs.array([[True] * n_samples]).transpose(1, 0)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_exp_log_vector(self):
        n_samples = 5
        expected = self.space_vector.random_uniform(n_samples)
        base_point = self.space_vector.random_uniform(n_samples)
        logs = self.space_vector.metric.log(expected, base_point)
        result = self.space_vector.metric.exp(logs, base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_exp_log_matrix(self):
        n_samples = 5
        expected = self.space_matrix.random_uniform(n_samples)
        base_point = self.space_matrix.random_uniform(n_samples)
        logs = self.space_matrix.metric.log(expected, base_point)
        result = self.space_matrix.metric.exp(logs, base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_dist_vector(self):
        n_samples = 5
        point = self.space_vector.random_uniform(n_samples)
        base_point = self.space_vector.random_uniform(n_samples)
        logs = self.space_vector.metric.log(point, base_point)
        logs = gs.einsum(
            '..., ...j->...j',
            1. / self.space_vector.metric.norm(logs, base_point),
            logs)
        point = self.space_vector.metric.exp(logs, base_point)
        result = self.space_vector.metric.dist(point, base_point)
        expected = gs.ones(n_samples)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_dist_matrix(self):
        n_samples = 5
        point = self.space_matrix.random_uniform(n_samples)
        base_point = self.space_matrix.random_uniform(n_samples)
        logs = self.space_matrix.metric.log(point, base_point)
        logs = gs.einsum(
            '...k, ...jl->...jl',
            1. / self.space_matrix.metric.norm(logs, base_point),
            logs)
        point = self.space_matrix.metric.exp(logs, base_point)
        result = self.space_matrix.metric.dist(point, base_point)
        expected = gs.ones((n_samples, 1))
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_inner_product_matrix_matrix(self):
        space = ProductManifold(
            manifolds=[Hypersphere(dimension=2).embedding_manifold,
                       Hyperbolic(dimension=2).embedding_manifold],
            default_point_type='matrix')
        point = space.random_uniform(1)
        result = space.metric.inner_product_matrix(point)
        expected = gs.identity(6)
        expected[3, 3] = - 1
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_inner_product_matrix_vector(self):
        space = ProductManifold(
            manifolds=[Hypersphere(dimension=2).embedding_manifold,
                       Hyperbolic(dimension=2).embedding_manifold],
            default_point_type='vector')
        point = space.random_uniform(1)
        expected = gs.identity(6)
        expected[3, 3] = - 1
        result = space.metric.inner_product_matrix(point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_regularize_vector(self):
        expected = self.space_vector.random_uniform(5)
        result = self.space_vector.regularize(expected)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_regularize_matrix(self):
        expected = self.space_matrix.random_uniform(5)
        result = self.space_matrix.regularize(expected)
        self.assertAllClose(result, expected)
