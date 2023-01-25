"""Unit tests for the Poincare ball."""

import pytest

import geomstats.backend as gs
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_ball import PoincareBall
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
        result = space.belongs(gs.array(point))
        self.assertAllClose(result, gs.array(expected))

    def test_projection_norm_lessthan_1(self, dim, point):
        space = self.Space(dim)
        projected_point = space.projection(gs.array(point))
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

    def test_mobius_out_of_the_ball(self, dim, x, y):
        metric = self.Metric(dim)
        with pytest.raises(ValueError):
            metric.mobius_add(gs.array(x), gs.array(y), project_first=False)

    def test_log(self, dim, point, base_point, expected):
        metric = self.Metric(dim)
        result = metric.log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_dist_pairwise(self, dim, point, expected):
        metric = self.Metric(dim)
        result = metric.dist_pairwise(gs.array(point))
        self.assertAllClose(result, gs.array(expected), rtol=1e-3)

    def test_dist(self, dim, point_a, point_b, expected):
        metric = self.Metric(dim)
        result = metric.dist(gs.array(point_a), gs.array(point_b))
        self.assertAllClose(result, gs.array(expected))

    def test_coordinate(self, dim, point_a, point_b):
        metric = self.Metric(dim)
        point_a_h = PoincareBall(dim).to_coordinates(gs.array(point_a), "extrinsic")
        point_b_h = PoincareBall(dim).to_coordinates(gs.array(point_b), "extrinsic")
        dist_in_ball = metric.dist(gs.array(point_a), gs.array(point_b))
        dist_in_hype = Hyperboloid(dim).metric.dist(point_a_h, point_b_h)
        self.assertAllClose(dist_in_ball, dist_in_hype)

    def test_mobius_vectorization(self, dim, point_a, point_b):
        metric = self.Metric(dim)

        dist_a_b = metric.mobius_add(point_a, point_b)

        result_vect = dist_a_b
        result = [metric.mobius_add(point_a, point_b[i]) for i in range(len(point_b))]
        result = gs.stack(result, axis=0)
        self.assertAllClose(result_vect, result)

        dist_a_b = metric.mobius_add(point_b, point_a)

        result_vect = dist_a_b
        result = [metric.mobius_add(point_b[i], point_a) for i in range(len(point_b))]
        result = gs.stack(result, axis=0)
        self.assertAllClose(result_vect, result)

    def test_log_vectorization(self, dim, point_a, point_b):

        metric = self.Metric(dim)
        dist_a_b = metric.log(point_a, point_b)

        result_vect = dist_a_b
        result = [metric.log(point_a, point_b[i]) for i in range(len(point_b))]
        result = gs.stack(result, axis=0)
        self.assertAllClose(result_vect, result)

        dist_a_b = metric.log(point_b, point_a)

        result_vect = dist_a_b
        result = [metric.log(point_b[i], point_a) for i in range(len(point_b))]
        result = gs.stack(result, axis=0)
        self.assertAllClose(result_vect, result)

    def test_exp_vectorization(self, dim, point_a, point_b):

        metric = self.Metric(dim)
        dist_a_b = metric.exp(point_a, point_b)

        result_vect = dist_a_b
        result = [metric.exp(point_a, point_b[i]) for i in range(len(point_b))]
        result = gs.stack(result, axis=0)
        self.assertAllClose(result_vect, result)

        dist_a_b = metric.exp(point_b, point_a)

        result_vect = dist_a_b
        result = [metric.exp(point_b[i], point_a) for i in range(len(point_b))]
        result = gs.stack(result, axis=0)
        self.assertAllClose(result_vect, result)
