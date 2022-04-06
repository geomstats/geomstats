"""Unit tests for ProductManifold."""
import random

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.product_manifold import (
    NFoldManifold,
    NFoldMetric,
    ProductManifold,
)
from geomstats.geometry.product_riemannian_metric import ProductRiemannianMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import Parametrizer
from tests.data_generation import _ManifoldTestData, _RiemannianMetricTestData
from tests.geometry_test_cases import ManifoldTestCase, RiemannianMetricTestCase

smoke_manifolds_1 = [Hypersphere(dim=2), Hyperboloid(dim=2)]
smoke_metrics_1 = [Hypersphere(dim=2).metric, Hyperboloid(dim=2).metric]

smoke_manifolds_2 = [Euclidean(3), Minkowski(3)]
smoke_metrics_2 = [Euclidean(3).metric, Minkowski(3).metric]


class TestProductManifold(ManifoldTestCase, metaclass=Parametrizer):
    space = ProductManifold
    skip_test_random_tangent_vec_is_tangent = True
    skip_test_projection_belongs = True

    class ProductManifoldTestData(_ManifoldTestData):

        n_list = random.sample(range(2, 4), 2)
        default_point_list = ["vector", "matrix"]
        manifolds_list = [[Hypersphere(dim=n), Hyperboloid(dim=n)] for n in n_list]
        space_args_list = [
            (manifold, None, default_point)
            for manifold, default_point in zip(manifolds_list, default_point_list)
        ]
        shape_list = [
            (n + 1, n + 1) if default_point == "matrix" else (2 * (n + 1),)
            for n, default_point in zip(n_list, default_point_list)
        ]
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def dimension_test_data(self):
            smoke_data = [
                dict(
                    manifold=smoke_manifolds_1,
                    default_point_type="vector",
                    expected=4,
                ),
                dict(
                    manifold=smoke_manifolds_1,
                    default_point_type="matrix",
                    expected=4,
                ),
            ]
            return self.generate_tests(smoke_data)

        def regularize_test_data(self):
            smoke_data = [
                dict(
                    manifold=smoke_manifolds_1,
                    default_point_type="vector",
                    point=ProductManifold(
                        smoke_manifolds_1, default_point_type="vector"
                    ).random_point(5),
                ),
                dict(
                    manifold=smoke_manifolds_1,
                    default_point_type="matrix",
                    point=ProductManifold(
                        smoke_manifolds_1, default_point_type="matrix"
                    ).random_point(5),
                ),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [
                (smoke_manifolds_1, None, "vector"),
                (smoke_manifolds_1, None, "matrix"),
            ]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=1e-1,
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                ProductManifold,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
                is_tangent_atol=gs.atol * 100,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                ProductManifold,
                self.space_args_list,
                self.n_vecs_list,
                is_tangent_atol=gs.atol * 100,
            )

    testing_data = ProductManifoldTestData()

    def test_dimension(self, manifolds, default_point_type, expected):
        space = self.space(manifolds, default_point_type=default_point_type)
        self.assertAllClose(space.dim, expected)

    def test_regularize(self, manifolds, default_point_type, point):
        space = self.space(manifolds, default_point_type=default_point_type)
        result = space.regularize(point)
        self.assertAllClose(result, point)


class TestProductRiemannianMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = ProductRiemannianMetric
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True

    class ProductRiemannianMetricTestData(_RiemannianMetricTestData):
        n_list = random.sample(range(2, 3), 1)
        default_point_list = ["vector", "matrix"]
        manifolds_list = [[Hypersphere(dim=n), Hyperboloid(dim=n)] for n in n_list]
        metrics_list = [
            [Hypersphere(dim=n).metric, Hyperboloid(dim=n).metric] for n in n_list
        ]
        metric_args_list = list(zip(metrics_list, default_point_list))
        shape_list = [
            (n + 1, n + 1) if default_point == "matrix" else (2 * (n + 1),)
            for n, default_point in zip(n_list, default_point_list)
        ]
        space_list = [
            ProductManifold(manifolds, None, default_point_type)
            for manifolds, default_point_type in zip(manifolds_list, default_point_list)
        ]
        n_points_list = random.sample(range(2, 5), 1)
        n_tangent_vecs_list = random.sample(range(2, 5), 1)
        n_points_a_list = random.sample(range(2, 5), 1)
        n_points_b_list = [1]
        alpha_list = [1] * 1
        n_rungs_list = [1] * 1
        scheme_list = ["pole"] * 1

        def inner_product_matrix_test_data(self):
            smoke_data = [
                dict(
                    metric=smoke_metrics_2,
                    default_point_type="vector",
                    point=ProductManifold(
                        smoke_manifolds_1, default_point_type="vector"
                    ).random_point(5),
                    base_point=ProductManifold(
                        smoke_manifolds_1, default_point_type="vector"
                    ).random_point(5),
                ),
                dict(
                    manifold=smoke_metrics_2,
                    default_point_type="matrix",
                    point=ProductManifold(
                        smoke_manifolds_2, default_point_type="matrix"
                    ).random_point(5),
                    base_point=ProductManifold(
                        smoke_manifolds_2, default_point_type="matrix"
                    ).random_point(5),
                ),
            ]
            return self.generate_tests(smoke_data)

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(self.metric_args_list, self.space_list)

        def squared_dist_is_symmetric_test_data(self):
            return self._squared_dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
                atol=gs.atol * 1000,
            )

        def exp_belongs_test_data(self):
            return self._exp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_is_tangent_test_data(self):
            return self._log_is_tangent_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                is_tangent_atol=1e-1,
            )

        def geodesic_ivp_belongs_test_data(self):
            return self._geodesic_ivp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_then_exp_test_data(self):
            return self._log_then_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 1000,
                atol=1e-1,
            )

        def exp_then_log_test_data(self):
            return self._exp_then_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                amplitude=10,
                rtol=gs.rtol * 1000,
                atol=1e-1,
            )

        def exp_ladder_parallel_transport_test_data(self):
            return self._exp_ladder_parallel_transport_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_rungs_list,
                self.alpha_list,
                self.scheme_list,
            )

        def exp_geodesic_ivp_test_data(self):
            return self._exp_geodesic_ivp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_points_list,
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
            )

        def parallel_transport_ivp_is_isometry_test_data(self):
            return self._parallel_transport_ivp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_test_data(self):
            return self._parallel_transport_bvp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def dist_is_symmetric_test_data(self):
            return self._dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_positive_test_data(self):
            return self._dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def squared_dist_is_positive_test_data(self):
            return self._squared_dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_norm_of_log_test_data(self):
            return self._dist_is_norm_of_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_point_to_itself_is_zero_test_data(self):
            return self._dist_point_to_itself_is_zero_test_data(
                self.metric_args_list, self.space_list, self.n_points_list
            )

        def inner_product_is_symmetric_test_data(self):
            return self._inner_product_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
            )

        def inner_product_matrix_vector_test_data(self):
            random_data = [
                dict(default_point_type="matrix"),
                dict(default_point_type="vector"),
            ]
            return self.generate_tests([], random_data)

        def dist_log_then_exp_norm_test_data(self):
            smoke_data = [
                dict(
                    space=smoke_manifolds_1,
                    default_point_type="vector",
                    n_samples=10,
                    einsum_str="..., ...j->...j",
                    expected=gs.ones(10),
                ),
                dict(
                    space=smoke_manifolds_1,
                    default_point_type="matrix",
                    n_samples=10,
                    einsum_str="..., ...jl->...jl",
                    expected=gs.ones(
                        10,
                    ),
                ),
            ]
            return self.generate_tests(smoke_data)

    testing_data = ProductRiemannianMetricTestData()

    @geomstats.tests.np_autograd_and_torch_only
    def test_inner_product_matrix(
        self, manifolds, default_point_type, point, base_point
    ):
        metric = self.metric(manifolds, default_point_type=default_point_type)
        logs = metric.log(point, base_point)
        result = metric.inner_product(logs, logs)
        expected = metric.squared_dist(base_point, point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_inner_product_matrix_vector(self, default_point_type):
        euclidean = Euclidean(3)
        minkowski = Minkowski(3)
        space = ProductManifold(manifolds=[euclidean, minkowski])
        point = space.random_point(1)
        expected = gs.eye(6)
        expected[3, 3] = -1
        result = space.metric.metric_matrix(point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_autograd_only
    def test_dist_log_then_exp_norm(
        self, manifolds, default_point_type, n_samples, einsum_str, expected
    ):
        space = ProductManifold(
            manifolds=manifolds, default_point_type=default_point_type
        )
        point = space.random_point(n_samples)
        base_point = space.random_point(n_samples)

        logs = space.metric.log(point, base_point)
        normalized_logs = gs.einsum(
            einsum_str,
            1.0 / space.metric.norm(logs, base_point),
            logs,
        )
        point = space.metric.exp(normalized_logs, base_point)
        result = space.metric.dist(point, base_point)
        self.assertAllClose(result, expected)


class TestNFoldManifold(ManifoldTestCase, metaclass=Parametrizer):
    space = NFoldManifold
    skip_test_random_tangent_vec_is_tangent = True

    class NFoldManifoldTestData(_ManifoldTestData):
        n_list = random.sample(range(2, 4), 2)
        base_list = [SpecialOrthogonal(n) for n in n_list]
        power_list = random.sample(range(2, 4), 2)
        space_args_list = list(zip(base_list, power_list))
        shape_list = [(power, n, n) for n, power in zip(n_list, power_list)]
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_test_data(self):
            smoke_data = [
                dict(
                    base=SpecialOrthogonal(3),
                    power=2,
                    point=gs.stack([gs.eye(3) + 1.0, gs.eye(3)])[None],
                    expected=gs.array(False),
                ),
                dict(
                    base=SpecialOrthogonal(3),
                    power=2,
                    point=gs.array([gs.eye(3), gs.eye(3)]),
                    expected=gs.array(True),
                ),
            ]
            return self.generate_tests(smoke_data)

        def shape_test_data(self):
            smoke_data = [dict(base=SpecialOrthogonal(3), power=2, shape=(2, 3, 3))]
            return self.generate_tests(smoke_data)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [
                (SpecialOrthogonal(2), 2),
                (SpecialOrthogonal(2), 2),
            ]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=1e-1,
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                NFoldManifold,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
                is_tangent_atol=gs.atol * 1000,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                NFoldManifold, self.space_args_list, self.n_vecs_list
            )

    def test_belongs(self, base, power, point, expected):
        space = self.space(base, power)
        self.assertAllClose(space.belongs(point), expected)

    def test_shape(self, base, power, expected):
        space = self.space(base, power)
        self.assertAllClose(space.shape, expected)

    testing_data = NFoldManifoldTestData()


class TestNFoldMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = NFoldMetric
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_log_is_tangent = True

    class NFoldMetricTestData(_RiemannianMetricTestData):

        n_list = random.sample(range(3, 5), 2)
        power_list = random.sample(range(2, 5), 2)
        base_list = [SpecialOrthogonal(n) for n in n_list]
        metric_args_list = [
            (base.metric, power) for base, power in zip(base_list, power_list)
        ]
        shape_list = [(power, n, n) for n, power in zip(n_list, power_list)]
        space_list = [
            NFoldManifold(base, power) for base, power in zip(base_list, power_list)
        ]
        n_points_list = random.sample(range(2, 5), 2)
        n_tangent_vecs_list = random.sample(range(2, 5), 2)
        n_points_a_list = random.sample(range(2, 5), 2)
        n_points_b_list = [1]
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(self.metric_args_list, self.space_list)

        def squared_dist_is_symmetric_test_data(self):
            return self._squared_dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
                atol=gs.atol * 1000,
            )

        def exp_belongs_test_data(self):
            return self._exp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_is_tangent_test_data(self):
            return self._log_is_tangent_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                is_tangent_atol=1e-1,
            )

        def geodesic_ivp_belongs_test_data(self):
            return self._geodesic_ivp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 100000,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 100000,
            )

        def log_then_exp_test_data(self):
            return self._log_then_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 10000,
                atol=1e-1,
            )

        def exp_then_log_test_data(self):
            return self._exp_then_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                amplitude=10.0,
                rtol=gs.rtol * 10000,
                atol=1e-1,
            )

        def exp_ladder_parallel_transport_test_data(self):
            return self._exp_ladder_parallel_transport_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_rungs_list,
                self.alpha_list,
                self.scheme_list,
            )

        def exp_geodesic_ivp_test_data(self):
            return self._exp_geodesic_ivp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_points_list,
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
            )

        def parallel_transport_ivp_is_isometry_test_data(self):
            return self._parallel_transport_ivp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_test_data(self):
            return self._parallel_transport_bvp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def dist_is_symmetric_test_data(self):
            print()
            return self._dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_positive_test_data(self):
            return self._dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def squared_dist_is_positive_test_data(self):
            return self._squared_dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_norm_of_log_test_data(self):
            return self._dist_is_norm_of_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_point_to_itself_is_zero_test_data(self):
            return self._dist_point_to_itself_is_zero_test_data(
                self.metric_args_list, self.space_list, self.n_points_list
            )

        def inner_product_is_symmetric_test_data(self):
            return self._inner_product_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
            )

        def inner_product_shape_test_data(self):
            space = NFoldManifold(SpecialOrthogonal(3), 2)
            n_samples = 4
            point = gs.stack([gs.eye(3)] * space.n_copies * n_samples)
            point = gs.reshape(point, (n_samples, *space.shape))
            tangent_vec = space.to_tangent(gs.zeros((n_samples, *space.shape)), point)
            smoke_data = [
                dict(space=space, n_samples=4, point=point, tangent_vec=tangent_vec)
            ]
            return self.generate_tests(smoke_data)

    testing_data = NFoldMetricTestData()

    def test_inner_product_shape(self, space, n_samples, point, tangent_vec):
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
