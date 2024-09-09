import pytest

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator, get_random_times
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data


class _SolverTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)


class ExpSolverAgainstMetricTestCase(_SolverTestCase):
    """Exp solver against equipped space test case."""

    @pytest.mark.random
    def test_exp(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        res_ = self.exp_solver.exp(tangent_vec, base_point)
        res = self.space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_ivp(self, n_points, n_times, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        time = get_random_times(n_times)

        res_ = self.exp_solver.geodesic_ivp(tangent_vec, base_point)(time)
        res = self.space.metric.geodesic(base_point, initial_tangent_vec=tangent_vec)(
            time
        )
        self.assertAllClose(res, res_, atol=atol)


class ExpSolverComparisonTestCase(_SolverTestCase):
    """Exp solver against exp solver test case."""

    @pytest.mark.random
    def test_exp(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        res = self.exp_solver.exp(tangent_vec, base_point)
        res_ = self.cmp_exp_solver.exp(tangent_vec, base_point)

        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_ivp(self, n_points, n_times, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        time = get_random_times(n_times)

        res = self.exp_solver.geodesic_ivp(tangent_vec, base_point)(time)
        res_ = self.cmp_exp_solver.geodesic_ivp(tangent_vec, base_point)(time)

        self.assertAllClose(res, res_, atol=atol)


class ExpSolverTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def test_exp(self, tangent_vec, base_point, expected, atol):
        res = self.exp_solver.exp(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_exp_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.exp_solver.exp(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="repeat-0",
        )
        self._test_vectorization(vec_data)

    def test_geodesic_ivp(self, tangent_vec, base_point, time, expected, atol):
        res = self.exp_solver.geodesic_ivp(tangent_vec, base_point)(time)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_geodesic_ivp_vec(self, n_reps, n_times, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        time = get_random_times(n_times)

        expected = self.exp_solver.geodesic_ivp(tangent_vec, base_point)(time)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    time=time,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="repeat-0",
        )
        self._test_vectorization(vec_data)


class LogSolverAgainstMetricTestCase(_SolverTestCase):
    """Log solver against equipped space test case."""

    @pytest.mark.random
    def test_log(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        res_ = self.log_solver.log(end_point, base_point)
        res = self.space.metric.log(end_point, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_bvp(self, n_points, n_times, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        time = get_random_times(n_times)

        res_ = self.log_solver.geodesic_bvp(end_point, base_point)(time)
        res = self.space.metric.geodesic(base_point, end_point=end_point)(time)
        self.assertAllClose(res, res_, atol=atol)


class LogSolverComparisonTestCase(_SolverTestCase):
    """Log solver against log solver test case."""

    @pytest.mark.random
    def test_log(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        res = self.log_solver.log(end_point, base_point)
        res_ = self.cmp_log_solver.log(end_point, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_bvp(self, n_points, n_times, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        time = get_random_times(n_times)

        res = self.log_solver.geodesic_bvp(end_point, base_point)(time)
        res_ = self.cmp_log_solver.geodesic_bvp(end_point, base_point)(time)
        self.assertAllClose(res, res_, atol=atol)


class LogSolverTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def test_log(self, point, base_point, expected, atol):
        res = self.log_solver.log(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_log_known_tangent_vec(self, n_points, atol):
        # NB: assumes proper implementation of exp
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        end_point = self.space.metric.exp(tangent_vec, base_point)

        self.test_log(end_point, base_point, tangent_vec, atol)

    def test_geodesic_bvp(self, point, base_point, time, expected, atol):
        res = self.log_solver.geodesic_bvp(point, base_point)(time)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_geodesic_bvp_vec(self, n_reps, n_times, atol):
        base_point = self.data_generator.random_point()
        point = self.data_generator.random_point()
        time = get_random_times(n_times)

        expected = self.log_solver.geodesic_bvp(point, base_point)(time)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    point=point,
                    base_point=base_point,
                    time=time,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["point", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_geodesic_bvp_known_geod(self, n_points, n_times, atol):
        # NB: assumes proper implementation of geodesic_ivp
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        time = get_random_times(n_times)

        point = self.space.metric.exp(tangent_vec, base_point)

        geod_point = self.space.metric.geodesic(
            base_point, initial_tangent_vec=tangent_vec
        )(time)

        self.test_geodesic_bvp(point, base_point, time, geod_point, atol)


class ExpSolverTypeCheck(_SolverTestCase):
    @pytest.mark.type
    def test_exp_type(self, n_points):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        res = self.exp_solver.exp(tangent_vec, base_point)
        self.assertTrue(gs.is_array(res), f"Wrong type: {type(res)}")

    @pytest.mark.type
    def test_geodesic_ivp_type(self, n_points, n_times):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        time = get_random_times(n_times)

        res = self.exp_solver.geodesic_ivp(tangent_vec, base_point)(time)
        self.assertTrue(gs.is_array(res), f"Wrong type: {type(res)}")


class LogSolverTypeCheckTestCase(_SolverTestCase):
    @pytest.mark.type
    def test_log_type(self, n_points):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        res = self.log_solver.log(end_point, base_point)
        self.assertTrue(gs.is_array(res), f"Wrong type: {type(res)}")

    @pytest.mark.type
    def test_geodesic_bvp_type(self, n_points, n_times):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        time = get_random_times(n_times)

        res = self.log_solver.geodesic_bvp(end_point, base_point)(time)
        self.assertTrue(gs.is_array(res), f"Wrong type: {type(res)}")
