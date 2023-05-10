"""Unit tests for ProductManifold, ProductRiemannianMetric."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.product_manifold import ProductManifold
from tests.conftest import Parametrizer
from tests.data.product_manifold_data import (
    ProductManifoldTestData,
    ProductRiemannianMetricTestData,
)
from tests.geometry_test_cases import ManifoldTestCase, RiemannianMetricTestCase


class TestProductManifold(ManifoldTestCase, metaclass=Parametrizer):
    skip_test_projection_belongs = True

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

    def test_inner_product_matrix(self, space, n_points):
        space.equip_with_metric(self.Metric)

        point = space.random_point(n_points)
        base_point = space.random_point(n_points)

        logs = space.metric.log(point, base_point)
        result = space.metric.inner_product(logs, logs, base_point)
        expected = space.metric.squared_dist(base_point, point)
        self.assertAllClose(result, expected)

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
    def test_dist_exp_after_log_norm(self, space, n_samples, einsum_str, expected):
        space.equip_with_metric(self.Metric)
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
