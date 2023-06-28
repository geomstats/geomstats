"""Unit tests for the Poincare ball."""

import pytest

import geomstats.backend as gs
from geomstats.geometry.hyperboloid import Hyperboloid
from tests.conftest import Parametrizer
from tests.data.poincare_ball_data import (
    PoincareBallTestData,
    TestDataPoincareBallMetric,
)
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestPoincareBall(OpenSetTestCase, metaclass=Parametrizer):
    skip_test_projection_belongs = True

    testing_data = PoincareBallTestData()

    def test_belongs(self, dim, point, expected):
        space = self.Space(dim)
        result = space.belongs(point)
        self.assertAllClose(result, expected)

    def test_projection_norm_lessthan_1(self, dim, point):
        space = self.Space(dim)
        projected_point = space.projection(point)
        result = gs.sum(projected_point * projected_point) < 1.0
        self.assertTrue(result)


class TestPoincareBallMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_belongs = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = TestDataPoincareBallMetric()

    def test_mobius_out_of_the_ball(self, space, x, y):
        space.equip_with_metric(self.Metric)
        with pytest.raises(ValueError):
            space.metric.mobius_add(x, y, project_first=False)

    def test_log(self, space, point, base_point, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.log(point, base_point)
        self.assertAllClose(result, expected)

    def test_dist_pairwise(self, space, point, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.dist_pairwise(point)
        self.assertAllClose(result, expected, rtol=1e-3)

    def test_dist(self, space, point_a, point_b, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.dist(point_a, point_b)
        self.assertAllClose(result, expected)

    def test_coordinate(self, space, point_a, point_b):
        space.equip_with_metric(self.Metric)
        point_a_h = space.to_coordinates(point_a, "extrinsic")
        point_b_h = space.to_coordinates(point_b, "extrinsic")
        dist_in_ball = space.metric.dist(point_a, point_b)
        dist_in_hype = Hyperboloid(space.dim).metric.dist(point_a_h, point_b_h)
        self.assertAllClose(dist_in_ball, dist_in_hype)

    def test_mobius_vectorization(self, space, point_a, point_b):
        space.equip_with_metric(self.Metric)

        dist_a_b = space.metric.mobius_add(point_a, point_b)

        result_vect = dist_a_b
        result = [
            space.metric.mobius_add(point_a, point_b[i]) for i in range(len(point_b))
        ]
        result = gs.stack(result, axis=0)
        self.assertAllClose(result_vect, result)

        dist_a_b = space.metric.mobius_add(point_b, point_a)

        result_vect = dist_a_b
        result = [
            space.metric.mobius_add(point_b[i], point_a) for i in range(len(point_b))
        ]
        result = gs.stack(result, axis=0)
        self.assertAllClose(result_vect, result)

    def test_log_vectorization(self, space, point_a, point_b):
        space.equip_with_metric(self.Metric)

        dist_a_b = space.metric.log(point_a, point_b)

        result_vect = dist_a_b
        result = [space.metric.log(point_a, point_b[i]) for i in range(len(point_b))]
        result = gs.stack(result, axis=0)
        self.assertAllClose(result_vect, result)

        dist_a_b = space.metric.log(point_b, point_a)

        result_vect = dist_a_b
        result = [space.metric.log(point_b[i], point_a) for i in range(len(point_b))]
        result = gs.stack(result, axis=0)
        self.assertAllClose(result_vect, result)

    def test_exp_vectorization(self, space, point_a, point_b):
        space.equip_with_metric(self.Metric)

        dist_a_b = space.metric.exp(point_a, point_b)

        result_vect = dist_a_b
        result = [space.metric.exp(point_a, point_b[i]) for i in range(len(point_b))]
        result = gs.stack(result, axis=0)
        self.assertAllClose(result_vect, result)

        dist_a_b = space.metric.exp(point_b, point_a)

        result_vect = dist_a_b
        result = [space.metric.exp(point_b[i], point_a) for i in range(len(point_b))]
        result = gs.stack(result, axis=0)
        self.assertAllClose(result_vect, result)
