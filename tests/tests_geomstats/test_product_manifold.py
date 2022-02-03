"""Unit tests for ProductManifold."""

import itertools

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.product_manifold import NFoldManifold, ProductManifold
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import Parametrizer, TestCase, TestData

space_matrix = ProductManifold(
    manifolds=[Hypersphere(dim=2), Hyperboloid(dim=2)],
    default_point_type="matrix",
)
space_vector = ProductManifold(
    manifolds=[Hypersphere(dim=2), Hyperboloid(dim=3)], default_point_type="vector"
)

manifolds = [space_matrix, space_vector]

SO3_2 = NFoldManifold(SpecialOrthogonal(3), 2)

n_samples = 4
SO3_2_point = gs.stack([gs.eye(3)] * SO3_2.n_copies * n_samples)
SO3_2_point = gs.reshape(SO3_2_point, (n_samples, *SO3_2.shape))
SO3_2_tangent_vec = SO3_2.to_tangent(gs.zeros((n_samples, *SO3_2.shape)), SO3_2_point)


class TestProductManifold(TestCase, metaclass=Parametrizer):
    cls = ProductManifold

    class TestDataProductManifold(TestData):
        def dimension_data(self):

            smoke_data = [
                dict(
                    manifold=space_matrix,
                    expected=4,
                ),
                dict(
                    manifold=space_vector,
                    expected=5,
                ),
            ]
            return self.generate_tests(smoke_data)

        def random_and_belongs_data(self):
            n_samples = [1, 5, 10]
            smoke_data = [
                dict(manifold=manifold, n_samples=n_samples)
                for (manifold, n_samples) in itertools.product(manifolds, n_samples)
            ]
            return self.generate_tests(smoke_data)

        def exp_log_data(self):
            n_samples = [5, 10]
            smoke_data = [
                dict(manifold=manifold, n_samples=n_samples)
                for (manifold, n_samples) in itertools.product(manifolds, n_samples)
            ]
            return self.generate_tests(smoke_data)

        def dist_log_exp_norm_data(self):
            n_samples = [1, 5, 10]
            smoke_data = [
                dict(manifold=manifold, n_samples=n_samples)
                for (manifold, n_samples) in itertools.product(
                    [manifolds[0]], n_samples
                )
            ]
            return self.generate_tests(smoke_data)

        def inner_product_data(self):
            euclidean = Euclidean(3)
            minkowski = Minkowski(3)
            space = ProductManifold(
                manifolds=[euclidean, minkowski], default_point_type="matrix"
            )
            point = space.random_point(1)
            expected = gs.eye(6)
            expected[3, 3] = -1
            smoke_data = [dict(manifold=space, point=point, expected=expected)]
            return self.generate_tests(smoke_data)

        def to_tangent_is_tangent_data(self):
            smoke_data = [dict(manifold=space_matrix), dict(manifold=space_vector)]
            return self.generate_tests(smoke_data)

        def projection_and_belongs_vector_data(self):
            smoke_data = [
                dict(
                    manifold=space_matrix,
                    shape=(
                        2,
                        len(space_matrix.manifolds),
                        space_matrix.manifolds[0].dim + 1,
                    ),
                ),
                dict(
                    manifold=space_vector,
                    shape=(2, space_vector.dim + 2),
                ),
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataProductManifold()

    def test_dimension(self, manifold, expected):
        self.assertAllClose(manifold.dim, expected)

    def test_random_and_belongs(self, manifold, n_samples):
        data = manifold.random_point(n_samples)
        self.assertAllClose(gs.all(manifold.belongs(data)), True)

    def test_exp_log(self, manifold, n_samples):
        expected = manifold.random_point(n_samples)
        base_point = manifold.random_point(n_samples)
        logs = manifold.metric.log(expected, base_point)
        result = manifold.metric.exp(logs, base_point)
        self.assertAllClose(result, expected, atol=1e-5)

    @geomstats.tests.np_and_autograd_only
    def test_dist_log_exp_norm(self, manifold, n_samples):
        n_samples = 10
        point = manifold.random_point(n_samples)
        base_point = manifold.random_point(n_samples)
        logs = manifold.metric.log(point, base_point)
        normalized_logs = gs.einsum(
            "..., ...jl->...jl",
            1.0 / manifold.metric.norm(logs, base_point),
            logs,
        )
        point = manifold.metric.exp(normalized_logs, base_point)
        result = manifold.metric.dist(point, base_point)
        expected = gs.ones((n_samples,))
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_inner_product(self, manifold, point, expected):
        result = manifold.metric.metric_matrix(gs.array(point))
        self.assertAllClose(result, expected)

    def test_to_tangent_is_tangent(self, manifold):
        space = manifold
        result = gs.all(helper.test_to_tangent_is_tangent(space, atol=gs.atol))
        self.assertAllClose(result, gs.array(True))

    @geomstats.tests.np_autograd_and_torch_only
    def test_projection_and_belongs_vector(self, manifold, shape):
        result = helper.test_projection_and_belongs(manifold, shape, atol=gs.atol * 100)
        self.assertAllClose(gs.all(result), gs.array(True))


class TestNFoldManifold(TestCase, metaclass=Parametrizer):
    class TestDataNFoldManifold(TestData):
        def random_and_belongs_data(self):
            smoke_data = [
                dict(product=SO3_2, n_samples=1),
                dict(product=SO3_2, n_samples=5),
            ]
            return self.generate_tests(smoke_data)

        def to_tangent_is_tangent_data(self):
            smoke_data = [dict(product=SO3_2)]
            return self.generate_tests(smoke_data)

        def projection_and_belongs_data(self):
            smoke_data = [dict(product=SO3_2, shape=(3, 2, 3, 3))]
            return self.generate_tests(smoke_data)

        def shape_data(self):
            smoke_data = [dict(product=SO3_2, expected=(2, 3, 3))]
            return self.generate_tests(smoke_data)

        def log_data(self):
            zeros = gs.zeros_like(SO3_2_point)
            smoke_data = [
                dict(
                    product=SO3_2,
                    point=SO3_2_point,
                    base_point=SO3_2_point,
                    expected=zeros,
                ),
                dict(
                    product=SO3_2,
                    point=SO3_2_point,
                    base_point=SO3_2_point[0],
                    expected=zeros,
                ),
                dict(
                    product=SO3_2,
                    point=SO3_2_point[0],
                    base_point=SO3_2_point[0],
                    expected=zeros[0],
                ),
            ]
            return self.generate_tests(smoke_data)

        def exp_data(self):
            smoke_data = [
                dict(
                    product=SO3_2,
                    tangent_vec=SO3_2_tangent_vec,
                    base_point=SO3_2_point,
                    expected=SO3_2_point,
                ),
                dict(
                    product=SO3_2,
                    tangent_vec=SO3_2_tangent_vec,
                    base_point=SO3_2_point[0],
                    expected=SO3_2_point,
                ),
                dict(
                    product=SO3_2,
                    tangent_vec=SO3_2_tangent_vec[0],
                    base_point=SO3_2_point[0],
                    expected=SO3_2_point[0],
                ),
            ]
            return self.generate_tests(smoke_data)

        def inner_product_data(self):
            zeros = gs.zeros(4)
            smoke_data = [
                dict(
                    product=SO3_2,
                    tangent_vec_a=SO3_2_tangent_vec,
                    tangent_vec_b=SO3_2_tangent_vec,
                    base_point=SO3_2_point,
                    expected=zeros,
                ),
                dict(
                    product=SO3_2,
                    tangent_vec_a=SO3_2_tangent_vec,
                    tangent_vec_b=SO3_2_tangent_vec,
                    base_point=SO3_2_point[0],
                    expected=zeros,
                ),
                dict(
                    product=SO3_2,
                    tangent_vec_a=SO3_2_tangent_vec[0],
                    tangent_vec_b=SO3_2_tangent_vec,
                    base_point=SO3_2_point,
                    expected=zeros,
                ),
                dict(
                    product=SO3_2,
                    tangent_vec_a=SO3_2_tangent_vec[0],
                    tangent_vec_b=SO3_2_tangent_vec[0],
                    base_point=SO3_2_point,
                    expected=zeros[0],
                ),
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataNFoldManifold()

    def test_random_and_belongs(self, product, n_samples):
        result = gs.all(product.belongs(product.random_point(n_samples)))
        self.assertAllClose(result, gs.array(True))

    def test_to_tangent_is_tangent(self, product):
        result = gs.all(helper.test_to_tangent_is_tangent(product))
        self.assertAllClose(result, gs.array(True))

    def test_projection_and_belongs(self, product, shape):
        result = gs.all(
            helper.test_projection_and_belongs(product, shape=shape, atol=1e-4)
        )
        self.assertAllClose(result, gs.array(True))

    def test_shape(self, product, expected):
        self.assertAllClose(product.shape, expected)

    def test_inner_product(
        self, product, tangent_vec_a, tangent_vec_b, point, expected
    ):
        result = product.metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, product, point, base_point, expected):
        result = product.metric.log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_exp(self, product, tangent_vec, base_point, expected):
        result = product.metric.exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))
