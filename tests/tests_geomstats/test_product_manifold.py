"""Unit tests for ProductManifold, ProductRiemannianMetric."""

import math

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
    skip_test_exp_shape = True

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
        space.equip_with_metric()
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
    
    def test_exp(self, space):
        space.equip_with_metric()
        point = space.random_point()
        tangent_vec = space.random_point()

        points = space.metric.exp(tangent_vec, point)
        points = space.project_from_product(points)
        factors = space.factors
        results = [manifold.shape == point.shape for manifold,point in zip(factors,points)]
        expected = gs.ones(len(factors))
        self.assertAllClose(results, expected)

    def test_exp_vectorization(self, space, n_samples):
        space.equip_with_metric()
        point = space.random_point()
        point = gs.tile(point,(n_samples, 1))
        tangent_vec = space.random_point()
        tangent_vec = gs.tile(tangent_vec, (n_samples, 1))

        results = space.metric.exp(tangent_vec, point)
        result = results[0]
        expected = gs.tile(result, (n_samples, 1))

        self.assertAllClose(results, expected)

    def test_log(self, space):
        space.equip_with_metric()
        point = space.random_point()
        base_point = space.random_point()

        tangent_vecs = space.metric.log(point, base_point)
        tangent_vecs = space.project_from_product(tangent_vecs)
        factors = space.factors
        results = [manifold.shape == point.shape for manifold,point in zip(factors,tangent_vecs)]
        expected = gs.ones(len(factors))

        self.assertAllClose(results, expected)

    def test_log_vectorization(self, space, n_samples):
        space.equip_with_metric()
        point = space.random_point()
        point = gs.tile(point,(n_samples, 1))
        base_point = space.random_point()
        base_point = gs.tile(base_point, (n_samples, 1))

        results = space.metric.log(point, base_point)
        result = results[0]
        expected = gs.tile(result, (n_samples, 1))

        self.assertAllClose(results, expected)

    def test_dist_vectorization(self, space, n_samples):
        space.equip_with_metric()
        point_a = space.random_point()
        point_a = gs.tile(point_a,(n_samples, 1))
        point_b = space.random_point()
        point_b = gs.tile(point_b, (n_samples, 1))

        results = space.metric.dist(point_a, point_b)
        result = results[0]
        expected = gs.repeat(result, n_samples)

        self.assertAllClose(results, expected)
    
    def test_geodesic(self, space):

        space.equip_with_metric()
        point_a = space.random_point()
        point_b = space.random_point() 

        point = space.metric.geodesic(point_a, point_b)(1/2)
        result = math.prod(point.shape)
        expected = math.prod(space.shape)
        self.assertAllClose(result, expected)

    def test_geodesic_vectorization(self, space, n_samples):
        space.equip_with_metric()
        point_a = space.random_point()
        point_b = space.random_point()
        times = gs.repeat(1/2, n_samples)

        results = space.metric.geodesic(point_a, point_b)(times)
        result = results[0]
        expected = gs.tile(result, (n_samples, 1))

        self.assertAllClose(results, expected)
        
