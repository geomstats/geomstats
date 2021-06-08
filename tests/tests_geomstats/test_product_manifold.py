"""Unit tests for ProductManifold."""

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.product_manifold import ProductManifold


class TestProductManifold(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.space_matrix = ProductManifold(
            manifolds=[Hypersphere(dim=2), Hyperboloid(dim=2)],
            default_point_type='matrix')
        self.space_vector = ProductManifold(
            manifolds=[Hypersphere(dim=2), Hyperboloid(dim=3)],
            default_point_type='vector')

    def test_dimension(self):
        expected = 5
        result = self.space_vector.dim
        self.assertAllClose(result, expected)

    def test_random_and_belongs_matrix(self):
        n_samples = 1
        data = self.space_matrix.random_point(n_samples)
        result = self.space_matrix.belongs(data)
        self.assertTrue(result)

        n_samples = 5
        data = self.space_matrix.random_point(n_samples)
        result = self.space_matrix.belongs(data)
        expected = gs.array([True] * n_samples)
        self.assertAllClose(result, expected)

    def test_random_and_belongs_vector(self):
        n_samples = 5
        data = self.space_vector.random_point(n_samples)
        result = self.space_vector.belongs(data)
        expected = gs.array([True] * n_samples)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_exp_log_vector(self):
        n_samples = 5
        expected = self.space_vector.random_point(n_samples)
        base_point = self.space_vector.random_point(n_samples)
        logs = self.space_vector.metric.log(expected, base_point)
        result = self.space_vector.metric.exp(logs, base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_exp_log_matrix(self):
        n_samples = 5
        expected = self.space_matrix.random_point(n_samples)
        base_point = self.space_matrix.random_point(n_samples)
        logs = self.space_matrix.metric.log(expected, base_point)
        result = self.space_matrix.metric.exp(logs, base_point)
        self.assertAllClose(result, expected, atol=1e-5)

    @geomstats.tests.np_only
    def test_dist_log_exp_norm_vector(self):
        n_samples = 5
        point = self.space_vector.random_point(n_samples)
        base_point = self.space_vector.random_point(n_samples)

        logs = self.space_vector.metric.log(point, base_point)
        normalized_logs = gs.einsum(
            '..., ...j->...j',
            1. / self.space_vector.metric.norm(logs, base_point),
            logs)
        point = self.space_vector.metric.exp(normalized_logs, base_point)
        result = self.space_vector.metric.dist(point, base_point)

        expected = gs.ones(n_samples)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_dist_log_exp_norm_matrix(self):
        n_samples = 10
        point = self.space_matrix.random_point(n_samples)
        base_point = self.space_matrix.random_point(n_samples)
        logs = self.space_matrix.metric.log(point, base_point)
        normalized_logs = gs.einsum(
            '..., ...jl->...jl',
            1. / self.space_matrix.metric.norm(logs, base_point),
            logs)
        point = self.space_matrix.metric.exp(normalized_logs, base_point)
        result = self.space_matrix.metric.dist(point, base_point)
        expected = gs.ones((n_samples,))
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_inner_product_matrix_matrix(self):
        euclidean = Euclidean(3)
        minkowski = Minkowski(3)
        space = ProductManifold(
            manifolds=[euclidean, minkowski],
            default_point_type='matrix')
        point = space.random_point(1)
        result = space.metric.metric_matrix(point)
        expected = gs.eye(6)
        expected[3, 3] = - 1
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_inner_product_matrix_vector(self):
        euclidean = Euclidean(3)
        minkowski = Minkowski(3)
        space = ProductManifold(
            manifolds=[euclidean, minkowski],
            default_point_type='vector')
        point = space.random_point(1)
        expected = gs.eye(6)
        expected[3, 3] = - 1
        result = space.metric.metric_matrix(point)
        self.assertAllClose(result, expected)

    def test_regularize_vector(self):
        expected = self.space_vector.random_point(5)
        result = self.space_vector.regularize(expected)
        self.assertAllClose(result, expected)

    def test_regularize_matrix(self):
        expected = self.space_matrix.random_point(5)
        result = self.space_matrix.regularize(expected)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_inner_product_matrix(self):
        n_samples = 1
        expected = self.space_matrix.random_point(n_samples)
        base_point = self.space_matrix.random_point(n_samples)
        logs = self.space_matrix.metric.log(expected, base_point)
        result = self.space_matrix.metric.inner_product(logs, logs)
        expected = self.space_matrix.metric.squared_dist(base_point, expected)
        self.assertAllClose(result, expected)

        n_samples = 5
        expected = self.space_matrix.random_point(n_samples)
        base_point = self.space_matrix.random_point(n_samples)
        logs = self.space_matrix.metric.log(expected, base_point)
        result = self.space_matrix.metric.inner_product(logs, logs)
        expected = self.space_matrix.metric.squared_dist(base_point, expected)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_projection_and_belongs_vector(self):
        space = self.space_vector
        shape = (2, space.dim + 2)
        result = helper.test_projection_and_belongs(
            space, shape, atol=gs.atol * 100)
        for res in result:
            self.assertTrue(res)

    @geomstats.tests.np_and_pytorch_only
    def test_projection_and_belongs_matrix(self):
        space = self.space_matrix
        shape = (2, len(space.manifolds), space.manifolds[0].dim + 1)
        result = helper.test_projection_and_belongs(
            space, shape, atol=gs.atol * 100)
        for res in result:
            self.assertTrue(res)

    def test_to_tangent_is_tangent_vector(self):
        space = self.space_vector
        result = helper.test_to_tangent_is_tangent(space, atol=gs.atol)
        for res in result:
            self.assertTrue(res)

    def test_to_tangent_is_tangent_matrix(self):
        space = self.space_matrix
        result = helper.test_to_tangent_is_tangent(space, atol=gs.atol)
        for res in result:
            self.assertTrue(res)
