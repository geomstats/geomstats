import pytest

from geomstats.test.random import RandomDataGenerator, get_random_times
from geomstats.test.test_case import TestCase


class _SolverComparisonTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)


class ExpSolverComparisonTestCase(_SolverComparisonTestCase):
    @pytest.mark.random
    def test_exp(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.exp(tangent_vec, base_point)
        res_ = self.exp_solver.exp(self.space, tangent_vec, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_ivp(self, n_points, n_times, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        time = get_random_times(n_times)

        res = self.space.metric.geodesic(base_point, initial_tangent_vec=tangent_vec)(
            time
        )
        res_ = self.exp_solver.geodesic_ivp(self.space, tangent_vec, base_point)(time)
        self.assertAllClose(res, res_, atol=atol)


class LogSolverComparisonTestCase(_SolverComparisonTestCase):
    @pytest.mark.random
    def test_log(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        res = self.space.metric.log(end_point, base_point)
        res_ = self.log_solver.log(self.space, end_point, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_bvp(self, n_points, n_times, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        time = get_random_times(n_times)

        res = self.space.metric.geodesic(base_point, end_point=end_point)(time)
        res_ = self.log_solver.geodesic_bvp(self.space, end_point, base_point)(time)
        self.assertAllClose(res, res_, atol=atol)
