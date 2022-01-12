"""Unit tests for ProductManifold."""

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.product_manifold import NFoldManifold, ProductManifold
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class TestProductManifold(geomstats.tests.TestCase):
    def setup_method(self):
        gs.random.seed(1234)

        self.space_matrix = ProductManifold(
            manifolds=[Hypersphere(dim=2), Hyperboloid(dim=2)],
            default_point_type="matrix",
        )
        self.space_vector = ProductManifold(
            manifolds=[Hypersphere(dim=2), Hyperboloid(dim=3)],
            default_point_type="vector",
        )

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

    @geomstats.tests.np_and_autograd_only
    def test_exp_log_vector(self):
        n_samples = 5
        expected = self.space_vector.random_point(n_samples)
        base_point = self.space_vector.random_point(n_samples)
        logs = self.space_vector.metric.log(expected, base_point)
        result = self.space_vector.metric.exp(logs, base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_exp_log_matrix(self):
        n_samples = 5
        expected = self.space_matrix.random_point(n_samples)
        base_point = self.space_matrix.random_point(n_samples)
        logs = self.space_matrix.metric.log(expected, base_point)
        result = self.space_matrix.metric.exp(logs, base_point)
        self.assertAllClose(result, expected, atol=1e-5)

    @geomstats.tests.np_and_autograd_only
    def test_dist_log_exp_norm_vector(self):
        n_samples = 5
        point = self.space_vector.random_point(n_samples)
        base_point = self.space_vector.random_point(n_samples)

        logs = self.space_vector.metric.log(point, base_point)
        normalized_logs = gs.einsum(
            "..., ...j->...j",
            1.0 / self.space_vector.metric.norm(logs, base_point),
            logs,
        )
        point = self.space_vector.metric.exp(normalized_logs, base_point)
        result = self.space_vector.metric.dist(point, base_point)

        expected = gs.ones(n_samples)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_dist_log_exp_norm_matrix(self):
        n_samples = 10
        point = self.space_matrix.random_point(n_samples)
        base_point = self.space_matrix.random_point(n_samples)
        logs = self.space_matrix.metric.log(point, base_point)
        normalized_logs = gs.einsum(
            "..., ...jl->...jl",
            1.0 / self.space_matrix.metric.norm(logs, base_point),
            logs,
        )
        point = self.space_matrix.metric.exp(normalized_logs, base_point)
        result = self.space_matrix.metric.dist(point, base_point)
        expected = gs.ones((n_samples,))
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_inner_product_matrix_matrix(self):
        euclidean = Euclidean(3)
        minkowski = Minkowski(3)
        space = ProductManifold(
            manifolds=[euclidean, minkowski], default_point_type="matrix"
        )
        point = space.random_point(1)
        result = space.metric.metric_matrix(point)
        expected = gs.eye(6)
        expected[3, 3] = -1
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_inner_product_matrix_vector(self):
        euclidean = Euclidean(3)
        minkowski = Minkowski(3)
        space = ProductManifold(
            manifolds=[euclidean, minkowski], default_point_type="vector"
        )
        point = space.random_point(1)
        expected = gs.eye(6)
        expected[3, 3] = -1
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

    @geomstats.tests.np_autograd_and_torch_only
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

    @geomstats.tests.np_autograd_and_torch_only
    def test_projection_and_belongs_vector(self):
        space = self.space_vector
        shape = (2, space.dim + 2)
        result = helper.test_projection_and_belongs(space, shape, atol=gs.atol * 100)
        for res in result:
            self.assertTrue(res)

    @geomstats.tests.np_autograd_and_torch_only
    def test_projection_and_belongs_matrix(self):
        space = self.space_matrix
        shape = (2, len(space.manifolds), space.manifolds[0].dim + 1)
        result = helper.test_projection_and_belongs(space, shape, atol=gs.atol * 100)
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


class TestNFoldManifold(geomstats.tests.TestCase):
    def setup_method(self):
        gs.random.seed(123)
        power = 2
        base = SpecialOrthogonal(3)
        space = NFoldManifold(base, power)
        self.product = space
        n_samples = 4
        point = gs.stack([gs.eye(3)] * space.n_copies * n_samples)
        point = gs.reshape(point, (n_samples, *space.shape))
        tangent_vec = space.to_tangent(gs.zeros((n_samples, *space.shape)), point)
        self.point = point
        self.tangent_vec = tangent_vec
        self.n_samples = n_samples

    def test_random_and_belongs(self):
        points = self.product.random_point()
        result = self.product.belongs(points)
        self.assertTrue(result)

        points = self.product.random_point(5)
        result = self.product.belongs(points)
        self.assertTrue(gs.all(result))

        not_a_point = gs.stack([gs.eye(3) + 1.0, gs.eye(3)])
        points = gs.concatenate([points, not_a_point[None]])
        result = self.product.belongs(points)
        expected = gs.array([True] * 5 + [False])
        self.assertAllClose(result, expected)

    def test_to_tangent_is_tangent(self):
        result = helper.test_to_tangent_is_tangent(self.product)
        for res in result:
            self.assertTrue(res)

    def test_projection_and_belongs(self):
        result = helper.test_projection_and_belongs(
            self.product, shape=(3, 2, 3, 3), atol=1e-4
        )
        for res in result:
            self.assertTrue(res)

    def test_inner_product_shape(self):
        space = self.product
        n_samples = self.n_samples
        point = self.point
        tangent_vec = self.tangent_vec
        result = space.metric.inner_product(tangent_vec, tangent_vec, point)
        expected = gs.zeros(n_samples)
        self.assertAllClose(result, expected)

        point = point[0]
        result = space.metric.inner_product(tangent_vec, tangent_vec, point)
        expected = gs.zeros(n_samples)
        self.assertAllClose(result, expected)

        result = space.metric.inner_product(tangent_vec[0], tangent_vec, point)
        self.assertAllClose(result, expected)

        expected = 0.0
        result = space.metric.inner_product(tangent_vec[0], tangent_vec[0], point)
        self.assertAllClose(result, expected)

    def test_exp(self):
        space = self.product
        point = self.point
        tangent_vec = self.tangent_vec
        result = space.metric.exp(tangent_vec, point)
        expected = point
        self.assertAllClose(result, expected)

        result = space.metric.exp(tangent_vec, point[0])
        expected = point
        self.assertAllClose(result, expected)

        result = space.metric.exp(tangent_vec[0], point[0])
        expected = point[0]
        self.assertAllClose(result, expected)

    def test_log(self):
        space = self.product
        point = self.point
        result = space.metric.log(point, point)
        expected = gs.zeros_like(point)
        self.assertAllClose(result, expected)

        result = space.metric.log(point, point[0])
        self.assertAllClose(result, expected)

        result = space.metric.log(point[0], point[0])
        expected = expected[0]
        self.assertAllClose(result, expected)

    def test_shape(self):
        result = self.product.shape
        expected = (2, 3, 3)
        self.assertAllClose(result, expected)

        result = self.product.shape
        self.assertAllClose(result, expected)
