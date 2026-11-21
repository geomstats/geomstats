import geomstats.backend as gs
from geomstats.test.test_case import TestCase


class ExpSolverAgainstMetricTestCase(TestCase):
    """Exp solver against equipped space test case."""

    def test_exp(self, tangent_vec, base_point, atol=gs.atol):
        """Check exponential map is the one of the metric.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        res = self.exp_solver.exp(tangent_vec, base_point)
        expected = self.space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_geodesic_ivp(self, initial_point, initial_tangent_vec, time, atol=gs.atol):
        """Check geodesic is the one of the metric.

        The geodesic is defined by the initial value problem.

        Parameters
        ----------
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        initial_tangent_vec : array-like, shape=[..., *space.shape]
            Initial velocity of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        atol : float
            Absolute tolerance.
        """
        res = self.exp_solver.geodesic_ivp(initial_tangent_vec, initial_point)(time)
        expected = self.space.metric.geodesic(
            initial_point, initial_tangent_vec=initial_tangent_vec
        )(time)
        self.assertAllClose(res, expected, atol=atol)


class ExpSolverComparisonTestCase(TestCase):
    """Exp solver against exp solver test case."""

    def test_exp(self, tangent_vec, base_point, atol=gs.atol):
        """Check exponential map is the same for both solvers.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        res = self.exp_solver.exp(tangent_vec, base_point)
        res_ = self.cmp_exp_solver.exp(tangent_vec, base_point)
        self.assertAllClose(res, res_, atol=atol)

    def test_geodesic_ivp(self, initial_point, initial_tangent_vec, time, atol=gs.atol):
        """Check geodesic is the same for both solvers.

        The geodesic is defined by the initial value problem.

        Parameters
        ----------
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        initial_tangent_vec : array-like, shape=[..., *space.shape]
            Initial velocity of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        atol : float
            Absolute tolerance.
        """
        res = self.exp_solver.geodesic_ivp(initial_tangent_vec, initial_point)(time)
        res_ = self.cmp_exp_solver.geodesic_ivp(initial_tangent_vec, initial_point)(
            time
        )
        self.assertAllClose(res, res_, atol=atol)


class ExpSolverTestCase(TestCase):
    """Exp solver test case."""

    def test_exp(self, tangent_vec, base_point, expected, atol=gs.atol):
        """Check exponential map.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected end point.
        atol : float
            Absolute tolerance.
        """
        res = self.exp_solver.exp(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_geodesic_ivp(self, tangent_vec, base_point, time, expected, atol=gs.atol):
        """Check geodesic defined by the initial value problem.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Initial velocity of the geodesic.
        base_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        expected : array-like, shape=[..., n_times, *space.shape]
            Expected points of the geodesic.
        atol : float
            Absolute tolerance.
        """
        res = self.exp_solver.geodesic_ivp(tangent_vec, base_point)(time)
        self.assertAllClose(res, expected, atol=atol)


class LogSolverAgainstMetricTestCase(TestCase):
    """Log solver against equipped space test case."""

    def test_log(self, point, base_point, atol=gs.atol):
        """Check logarithm map is the one of the metric.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        res = self.log_solver.log(point, base_point)
        expected = self.space.metric.log(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_geodesic_bvp(self, initial_point, end_point, time, atol=gs.atol):
        """Check geodesic is the one of the metric.

        The geodesic is defined by the boundary value problem.

        Parameters
        ----------
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        end_point : array-like, shape=[..., *space.shape]
            End point of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        atol : float
            Absolute tolerance.
        """
        res = self.log_solver.geodesic_bvp(end_point, initial_point)(time)
        expected = self.space.metric.geodesic(initial_point, end_point=end_point)(time)
        self.assertAllClose(res, expected, atol=atol)


class LogSolverTestCase(TestCase):
    """Log solver test case."""

    def test_log(self, point, base_point, expected, atol=gs.atol):
        """Check logarithm map.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
        res = self.log_solver.log(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_geodesic_bvp(self, point, base_point, time, expected, atol=gs.atol):
        """Check geodesic defined by the boundary value problem.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            End point of the geodesic.
        base_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        expected : array-like, shape=[..., n_times, *space.shape]
            Expected points of the geodesic.
        atol : float
            Absolute tolerance.
        """
        res = self.log_solver.geodesic_bvp(point, base_point)(time)
        self.assertAllClose(res, expected, atol=atol)

    def test_log_after_exp(self, tangent_vec, base_point, atol=gs.atol):
        """Check logarithm map recovers the tangent vector shot by the metric.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.

        Notes
        -----
        Assumes the exponential map of the metric is correct.
        """
        point = self.space.metric.exp(tangent_vec, base_point)
        self.test_log(point, base_point, tangent_vec, atol=atol)

    def test_geodesic_bvp_known_geod(self, tangent_vec, base_point, time, atol=gs.atol):
        """Check geodesic is the one defined by the initial value problem.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Initial velocity of the geodesic.
        base_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        atol : float
            Absolute tolerance.

        Notes
        -----
        Assumes the exponential map and the geodesic defined by the initial
        value problem of the metric are correct.
        """
        point = self.space.metric.exp(tangent_vec, base_point)
        expected = self.space.metric.geodesic(
            base_point, initial_tangent_vec=tangent_vec
        )(time)
        self.test_geodesic_bvp(point, base_point, time, expected, atol=atol)
