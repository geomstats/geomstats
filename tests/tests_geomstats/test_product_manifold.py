"""Unit tests for ProductManifold."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.product_riemannian_metric import NFoldMetric
from tests.conftest import Parametrizer
from tests.data.product_manifold_data import (
    NFoldManifoldTestData,
    NFoldMetricTestData,
    ProductManifoldTestData,
    ProductRiemannianMetricTestData,
)
from tests.geometry_test_cases import ManifoldTestCase, RiemannianMetricTestCase


class TestProductManifold(ManifoldTestCase, metaclass=Parametrizer):

    testing_data = ProductManifoldTestData()

    def test_dimension(self, manifolds, default_point_type, expected):
        space = self.Space(manifolds, default_point_type=default_point_type)
        self.assertAllClose(space.dim, expected)

    def test_regularize(self, manifolds, default_point_type, point):
        space = self.Space(manifolds, default_point_type=default_point_type)
        result = space.regularize(point)
        self.assertAllClose(result, point)

    def test_default_coords_type(self, space_args, expected):
        space = self.Space(*space_args)
        self.assertTrue(
            space.default_coords_type == expected,
            msg=f"Expected `{expected}`, but it is `{space.default_coords_type}`",
        )

    def test_embed_to_after_project_from(self, space_args, n_points):
        space = self.Space(*space_args)

        points = space.random_point(n_points)

        factors_points = space.project_from_product(points)
        points_ = space.embed_to_product(factors_points)

        self.assertAllClose(points, points_)


class TestProductRiemannianMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True
    skip_test_inner_product_matrix_vector = True

    testing_data = ProductRiemannianMetricTestData()

    @tests.conftest.np_autograd_and_torch_only
    def test_inner_product_matrix(
        self, manifolds, default_point_type, point, base_point
    ):
        metric = self.Metric(manifolds, default_point_type=default_point_type)
        logs = metric.log(point, base_point)
        result = metric.inner_product(logs, logs, base_point)
        expected = metric.squared_dist(base_point, point)
        self.assertAllClose(result, expected)

    @tests.conftest.np_autograd_and_torch_only
    def test_inner_product_matrix_vector(self, default_point_type):
        euclidean = Euclidean(3)
        minkowski = Minkowski(3)
        space = ProductManifold(factors=[euclidean, minkowski])
        point = space.random_point(1)
        expected = gs.eye(6)
        expected[3, 3] = -1
        result = space.metric.metric_matrix(point)
        self.assertAllClose(result, expected)

    @tests.conftest.np_and_autograd_only
    def test_dist_exp_after_log_norm(
        self, manifolds, default_point_type, n_samples, einsum_str, expected
    ):
        space = ProductManifold(
            factors=manifolds, default_point_type=default_point_type
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
    testing_data = NFoldManifoldTestData()

    def test_belongs(self, base, power, point, expected):
        space = self.Space(base, power)
        self.assertAllEqual(space.belongs(point), expected)

    def test_shape(self, base, power, expected):
        space = self.Space(base, power)
        self.assertAllClose(space.shape, expected)


class TestNFoldMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

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

    @tests.conftest.np_autograd_and_torch_only
    def test_inner_product_scales(
        self, base_metric, n_copies, scales, point, tangent_vec
    ):
        metric = NFoldMetric(base_metric=base_metric, n_copies=n_copies, scales=scales)
        result = metric.inner_product(tangent_vec, tangent_vec, point)

        expected = 0
        base_shape = base_metric.shape
        point_reshaped = gs.reshape(point, (-1, n_copies, *base_shape))
        vec_reshaped = gs.reshape(tangent_vec, (-1, n_copies, *base_shape))
        for i in range(n_copies):
            point_i = point_reshaped[:, i]
            vec_i = vec_reshaped[:, i]
            expected += scales[i] * base_metric.inner_product(vec_i, vec_i, point_i)
        self.assertAllClose(result, expected)
