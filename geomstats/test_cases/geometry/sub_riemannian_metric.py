import geomstats.backend as gs
from geomstats.test.test_case import TestCase


class SubRiemannianMetricTestCase(TestCase):
    def test_inner_coproduct(
        self, cotangent_vec_a, cotangent_vec_b, base_point, expected, atol=gs.atol
    ):
        """Check inner coproduct of two cotangent vectors.

        Parameters
        ----------
        cotangent_vec_a : array-like, shape=[..., dim]
            Cotangent vector at base point.
        cotangent_vec_b : array-like, shape=[..., dim]
            Cotangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        expected : array-like, shape=[...,]
            Expected inner coproduct.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_hamiltonian(self, cotangent_vec, base_point, expected, atol=gs.atol):
        """Check Hamiltonian at the state given by a point and a cotangent vector.

        Parameters
        ----------
        cotangent_vec : array-like, shape=[dim]
            Cotangent vector at base point.
        base_point : array-like, shape=[dim]
            Point on the manifold.
        expected : float
            Expected Hamiltonian energy.
        atol : float
            Absolute tolerance.
        """
        state = gs.array([base_point, cotangent_vec])
        res = self.space.metric.hamiltonian(state)
        self.assertAllClose(res, expected, atol=atol)

    def test_sr_sharp(self, cotangent_vec, base_point, expected, atol=gs.atol):
        """Check sub-Riemannian sharp map of a cotangent vector.

        Parameters
        ----------
        cotangent_vec : array-like, shape=[..., dim]
            Cotangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        expected : array-like, shape=[..., dim]
            Expected tangent vector at base point.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.sr_sharp(base_point, cotangent_vec)
        self.assertAllClose(res, expected, atol=atol)

    def test_symp_grad(self, state, expected, atol=gs.atol):
        """Check symplectic gradient of the Hamiltonian.

        Parameters
        ----------
        state : array-like, shape=[2, dim]
            Point and cotangent vector.
        expected : array-like, shape=[2, dim]
            Expected symplectic gradient.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.symp_grad(hamiltonian=self.space.metric.hamiltonian)(
            state
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_symp_euler(self, state, step_size, expected, atol=gs.atol):
        """Check one step of symplectic Euler integration.

        Parameters
        ----------
        state : array-like, shape=[2, dim]
            Point and cotangent vector.
        step_size : float
            Step size.
        expected : array-like, shape=[2, dim]
            Expected state after one step.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.symp_euler(
            hamiltonian=self.space.metric.hamiltonian, step_size=step_size
        )(state)
        self.assertAllClose(res, expected, atol=atol)

    def test_iterate(self, state, n_steps, step_size, expected, atol=gs.atol):
        """Check last state of iterated symplectic Euler steps.

        Parameters
        ----------
        state : array-like, shape=[2, dim]
            Initial point and cotangent vector.
        n_steps : int
            Number of steps.
        step_size : float
            Step size.
        expected : array-like, shape=[2, dim]
            Expected state after `n_steps` steps.
        atol : float
            Absolute tolerance.
        """
        step = self.space.metric.symp_euler(
            hamiltonian=self.space.metric.hamiltonian, step_size=step_size
        )
        res = self.space.metric.iterate(step, n_steps)(state)[-1]
        self.assertAllClose(res, expected, atol=atol)

    def test_symp_flow(self, state, end_time, n_steps, expected, atol=gs.atol):
        """Check end state of the symplectic flow of the Hamiltonian.

        Parameters
        ----------
        state : array-like, shape=[2, dim]
            Initial point and cotangent vector.
        end_time : float
            Integration end time.
        n_steps : int
            Number of steps.
        expected : array-like, shape=[2, dim]
            Expected state at `end_time`.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.symp_flow(
            hamiltonian=self.space.metric.hamiltonian,
            end_time=end_time,
            n_steps=n_steps,
        )(state)[-1]
        self.assertAllClose(res, expected, atol=atol)

    def test_exp(self, cotangent_vec, base_point, n_steps, expected, atol=gs.atol):
        """Check exponential map of a cotangent vector.

        Parameters
        ----------
        cotangent_vec : array-like, shape=[..., dim]
            Cotangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_steps : int
            Number of integration steps.
        expected : array-like, shape=[..., dim]
            Expected point on the manifold.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.exp(cotangent_vec, base_point, n_steps=n_steps)
        self.assertAllClose(res, expected, atol=atol)

    def test_geodesic(
        self,
        initial_point,
        initial_cotangent_vec,
        times,
        n_steps,
        expected,
        atol=gs.atol,
    ):
        """Check normal geodesic at given times.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Initial point of the geodesic.
        initial_cotangent_vec : array-like, shape=[..., dim]
            Initial cotangent vector.
        times : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        n_steps : int
            Number of integration steps.
        expected : array-like, shape=[..., n_times, dim]
            Expected points on the geodesic.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.geodesic(
            initial_point=initial_point,
            initial_cotangent_vec=initial_cotangent_vec,
            n_steps=n_steps,
        )(times)
        self.assertAllClose(res, expected, atol=atol)
