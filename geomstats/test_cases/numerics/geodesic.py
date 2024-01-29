import pytest

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator, get_random_times
from geomstats.test.test_case import TestCase


class _SolverTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)


class ExpSolverComparisonTestCase(_SolverTestCase):
    @pytest.mark.random
    def test_exp(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        res_ = self.exp_solver.exp(self.space, tangent_vec, base_point)
        res = self.space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_ivp(self, n_points, n_times, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        time = get_random_times(n_times)

        res_ = self.exp_solver.geodesic_ivp(self.space, tangent_vec, base_point)(time)
        res = self.space.metric.geodesic(base_point, initial_tangent_vec=tangent_vec)(
            time
        )
        self.assertAllClose(res, res_, atol=atol)


class LogSolverComparisonTestCase(_SolverTestCase):
    @pytest.mark.random
    def test_log(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        res_ = self.log_solver.log(self.space, end_point, base_point)
        res = self.space.metric.log(end_point, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_bvp(self, n_points, n_times, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        time = get_random_times(n_times)

        res_ = self.log_solver.geodesic_bvp(self.space, end_point, base_point)(time)
        res = self.space.metric.geodesic(base_point, end_point=end_point)(time)
        self.assertAllClose(res, res_, atol=atol)


class ExpSolverTypeCheck(_SolverTestCase):
    @pytest.mark.type
    def test_exp_type(self, n_points):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        res = self.exp_solver.exp(self.space, tangent_vec, base_point)
        self.assertTrue(gs.is_array(res), f"Wrong type: {type(res)}")

    @pytest.mark.type
    def test_geodesic_ivp_type(self, n_points, n_times):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        time = get_random_times(n_times)

        res = self.exp_solver.geodesic_ivp(self.space, tangent_vec, base_point)(time)
        self.assertTrue(gs.is_array(res), f"Wrong type: {type(res)}")


class LogSolverTypeCheckTestCase(_SolverTestCase):
    @pytest.mark.random
    def test_log_type(self, n_points):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        res = self.log_solver.log(self.space, end_point, base_point)
        self.assertTrue(gs.is_array(res), f"Wrong type: {type(res)}")

    @pytest.mark.random
    def test_geodesic_bvp_type(self, n_points, n_times):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        time = get_random_times(n_times)

        res = self.log_solver.geodesic_bvp(self.space, end_point, base_point)(time)
        self.assertTrue(gs.is_array(res), f"Wrong type: {type(res)}")
