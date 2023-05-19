"""Unit tests for NFoldManifold, NFoldMetric."""

import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.nfold_manifold_data import NFoldManifoldTestData, NFoldMetricTestData
from tests.geometry_test_cases import ManifoldTestCase, RiemannianMetricTestCase


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
        space.equip_with_metric(self.Metric)

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

    def test_inner_product_scales(self, space, scales, point, tangent_vec):
        space.equip_with_metric(self.Metric, scales=scales)

        result = space.metric.inner_product(tangent_vec, tangent_vec, point)

        expected = 0
        base_shape = space.base_manifold.shape
        point_reshaped = gs.reshape(point, (-1, space.n_copies, *base_shape))
        vec_reshaped = gs.reshape(tangent_vec, (-1, space.n_copies, *base_shape))
        for i in range(space.n_copies):
            point_i = point_reshaped[:, i]
            vec_i = vec_reshaped[:, i]
            expected += scales[i] * space.base_manifold.metric.inner_product(
                vec_i, vec_i, point_i
            )
        self.assertAllClose(result, expected)
