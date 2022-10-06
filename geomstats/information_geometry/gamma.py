"""Statistical Manifold of Gamma distributions with the Fisher metric.

Lead author: Jules Deschamps.
"""
import logging
import warnings

import numpy as np
from scipy.integrate import odeint, solve_bvp
from scipy.optimize import minimize
from scipy.stats import gamma

import geomstats.backend as gs
import geomstats.errors
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.information_geometry.base import InformationManifoldMixin

warnings.filterwarnings("error")

N_STEPS = 100

"""
The natural coordinate system for a Gamma Distribution is:
point = [kappa, nu], where kappa is the shape parameter, and nu the rate, or 1/scale.

However, information geometry most often works with standard coordinates, given by:
point = [kappa, gamma] = [kappa, kappa/nu].

The standard coordinate system is the convention we use in this script.
All points and all vectors input are assumed to be given in the standard coordinate
system unless stated otherwise.

Some of the methods in GammaDistributions allow to easily make the associated
change of variable, either for a point or a vector.
"""


class GammaDistributions(InformationManifoldMixin, OpenSet):
    """Class for the manifold of Gamma distributions.

    This is :math: Gamma = `(R_+^*)^2`, the positive quadrant of the
    2-dimensional Euclidean space.
    """

    def __init__(self):
        super().__init__(dim=2, ambient_space=Euclidean(2), metric=GammaMetric())

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold of Gamma distributions.

        Check that point defines parameters for a Gamma distribution,
        i.e. belongs to the positive quadrant of the Euclidean space.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point to be checked.
        atol : float
            Tolerance to evaluate positivity.
            Optional, default: gs.atol

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean indicating whether point represents a Gamma
            distribution.
        """
        point_dim = point.shape[-1]
        belongs = point_dim == 2
        belongs = gs.logical_and(belongs, gs.all(point >= atol, axis=-1))
        return belongs

    def random_point(self, n_samples=1, upper_bound=5.0, lower_bound=0.0):
        """Sample parameters of Gamma distributions.

        The uniform distribution on [0, bound]^2 is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of the square where the Gamma parameters are sampled.
            Optional, default: 5.

        Returns
        -------
        samples : array-like, shape=[..., 2]
            Sample of points representing Gamma distributions.
        """
        upper_bound, lower_bound = gs.array(upper_bound) * gs.ones(2), gs.array(
            lower_bound
        ) * gs.ones(2)

        if gs.any((upper_bound - lower_bound) < 0):
            raise ValueError("upper_bound cannot be greater than lower_bound.")

        size = (2,) if n_samples == 1 else (n_samples, 2)
        return lower_bound + (upper_bound - lower_bound) * gs.random.rand(*size)

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the open set.

        The last coordinate is floored to `gs.atol` if it is negative.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point in ambient space.
        atol : float
            Tolerance to evaluate positivity.

        Returns
        -------
        projected : array-like, shape=[..., 2]
            Projected point.
        """
        return gs.where(point < atol, atol, point)

    def sample(self, point, n_samples=1):
        """Sample from the Gamma distribution.

        Sample from the Gamma distribution with parameters provided
        by point. This gives n_samples points.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a Gamma distribution.
        n_samples : int
            Number of points to sample for each set of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from the Gamma distributions.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=2)
        samples = []
        for param in point:
            sample = gs.array(
                gamma.rvs(param[0], loc=0, scale=param[1] / param[0], size=n_samples)
            )
            samples.append(sample)
        return gs.squeeze(samples[0]) if len(point) == 1 else gs.stack(samples)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the Gamma
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a Gamma distribution.

        Returns
        -------
        pdf : function
            Probability density function of the Gamma distribution with
            parameters provided by point.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=2)

        def pdf(x):
            """Generate parameterized function for Gamma pdf.

            Parameters
            ----------
            x : array-like, shape=[n_points, dim]
                Points at which to compute the probability
                density function.

            Returns
            -------
            pdf_at_x : array-like, shape=[..., n_points]
                Values of pdf at x for each value of the parameters provided
                by point.
            """
            pdf_at_x = gs.array(
                [
                    gamma.pdf(t, a=point[:, 0], scale=point[:, 1] / point[:, 0])
                    for t in gs.array(x)
                ]
            )
            pdf_at_x = gs.squeeze(gs.stack(pdf_at_x, axis=0)).transpose()

            return pdf_at_x

        return pdf

    @staticmethod
    def maximum_likelihood_fit(data):
        """Estimate parameters from samples.

        This is a wrapper around scipy's maximum likelihood estimator to
        estimate the parameters of a gamma distribution from samples.

        Parameters
        ----------
        data : list or list of lists/arrays
            Data to estimate parameters from. Lists of
            different length may be passed.

        Returns
        -------
        parameter : array-like, shape=[..., 2]
            Estimate of parameter obtained by maximum likelihood.
        """

        def is_nested(sample):
            """Check if sample contains an iterable."""
            for el in sample:
                try:
                    return iter(el)
                except TypeError:
                    return False

        if not is_nested(data):
            data = [data]
        parameters = []
        for sample in data:
            sample = gs.array(sample)
            kappa, _, scale = gamma.fit(sample, floc=0)
            nu = 1 / scale
            parameters.append(gs.array([kappa, kappa / nu]))
        return parameters[0] if len(data) == 1 else gs.stack(parameters)

    def natural_to_standard(self, point):
        """Convert point from natural coordinates to standard coordinates.

        The change of variable is symmetric.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point of the Gamma manifold, given in natural coordinates.

        Returns
        -------
        point : array-like, shape=[..., 2]
            Point of the Gamma manifold, given in standard coordinates.
        """
        point = gs.to_ndarray(point, to_ndim=2)

        point = gs.transpose(gs.stack([point[:, 0], point[:, 0] / point[:, 1]]))

        return gs.squeeze(point)

    def standard_to_natural(self, point):
        """Convert point from standard coordinates to natural coordinates.

        The change of variable is symmetric.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point of the Gamma manifold, given in standard coordinates.

        Returns
        -------
        point : array-like, shape=[..., 2]
            Point of the Gamma manifold, given in natural coordinates.
        """
        return self.natural_to_standard(point)

    def tangent_natural_to_standard(self, vec, base_point):
        """Convert tangent vector from natural coordinates to standard coordinates.

        The change of variable is symmetric.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point of the Gamma manifold, given in natural coordinates.
        vec : array-like, shape=[..., 2]
            Tangent vector at base_point, given in natural coordinates.

        Returns
        -------
        vec : array-like, shape=[..., 2]
            Tangent vector at base_point, given in standard coordinates.
        """
        vec = gs.to_ndarray(vec, to_ndim=2)
        base_point = gs.broadcast_to(base_point, vec.shape)

        n_points = base_point.shape[0]

        kappa, scale = (
            base_point[..., 0],
            base_point[..., 1],
        )

        jac = gs.array(
            [[gs.ones(n_points), gs.zeros(n_points)], [1 / scale, -kappa / scale**2]]
        )
        jac = gs.transpose(jac, [2, 0, 1])

        vec = gs.einsum("...jk,...k->...j", jac, vec)

        return gs.squeeze(vec)

    def tangent_standard_to_natural(self, vec, base_point):
        """Convert tangent vector from standard coordinates to natural coordinates.

        The change of variable is symmetric.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point of the Gamma manifold, given in standard coordinates.
        vec : array-like, shape=[..., 2]
            Tangent vector at base_point, given in standard coordinates.

        Returns
        -------
        vec : array-like, shape=[..., 2]
            Tangent vector at base_point, given in natural coordinates.
        """
        return self.tangent_natural_to_standard(vec, base_point)


class GammaMetric(RiemannianMetric):
    """Class for the Fisher information metric on Gamma distributions.

    References
    ----------
    .. [AD2008] Arwini, K. A., & Dodson, C. T. (2008).
        Information geometry (pp. 31-54). Springer Berlin Heidelberg.
    """

    def __init__(self):
        super().__init__(dim=2)

    def metric_matrix(self, base_point=None):
        """Compute the inner-product matrix.

        Compute the inner-product matrix of the Fisher information metric
        at the tangent space at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., 2, 2]
            Inner-product matrix.
        """
        if base_point is None:
            raise ValueError(
                "A base point must be given to compute the " "metric matrix"
            )
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        kappa, gamma = base_point[:, 0], base_point[:, 1]

        mat_diag = gs.transpose(
            gs.array([gs.polygamma(1, kappa) - 1 / kappa, kappa / gamma**2])
        )
        mat = from_vector_to_diagonal_matrix(mat_diag)
        return gs.squeeze(mat)

    def christoffels(self, base_point):
        """Compute the Christoffel symbols.

        Compute the Christoffel symbols of the Fisher information metric.
        For computation purposes, we replace the value of
        (gs.polygamma(1, x) - 1/x) by an equivalent (close lower-bound) when it becomes
        too difficult to compute, as per in the second reference.

        References
        ----------
        .. [AD2008] Arwini, K. A., & Dodson, C. T. (2008).
            Information geometry (pp. 31-54). Springer Berlin Heidelberg.

        .. [GQ2015] Guo, B. N., Qi, F., Zhao, J. L., & Luo, Q. M. (2015).
            Sharp inequalities for polygamma functions.
            Mathematica Slovaca, 65(1), 103-120.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        christoffels : array-like, shape=[..., 2, 2, 2]
            Christoffel symbols, with the contravariant index on
            the first dimension.
            :math: 'christoffels[..., i, j, k] = Gamma^i_{jk}'
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        kappa, gamma = base_point[:, 0], base_point[:, 1]

        if gs.any(kappa > 4e15):
            raise ValueError(
                "Christoffels computation overflows with values of kappa. "
                "All values of kappa < 4e15 work."
            )

        shape = kappa.shape

        c111 = gs.where(
            gs.polygamma(1, kappa) - 1 / kappa > gs.atol,
            (gs.polygamma(2, kappa) + gs.array(kappa) ** -2)
            / (2 * (gs.polygamma(1, kappa) - 1 / kappa)),
            0.25 * (kappa**2 * gs.polygamma(2, kappa) + 1),
        )

        c122 = gs.where(
            gs.polygamma(1, kappa) - 1 / kappa > gs.atol,
            -1 / (2 * gamma**2 * (gs.polygamma(1, kappa) - 1 / kappa)),
            -(kappa**2) / (4 * gamma**2),
        )

        c1 = gs.squeeze(
            from_vector_to_diagonal_matrix(gs.transpose(gs.array([c111, c122])))
        )

        c2 = gs.squeeze(
            gs.transpose(
                gs.array(
                    [[gs.zeros(shape), 1 / (2 * kappa)], [1 / (2 * kappa), -1 / gamma]]
                )
            )
        )

        christoffels = gs.array([c1, c2])

        if len(christoffels.shape) == 4:
            christoffels = gs.transpose(christoffels, [1, 0, 2, 3])

        return gs.squeeze(christoffels)

    def jacobian_christoffels(self, base_point):
        """Compute the Jacobian of the Christoffel symbols.

        Compute the Jacobian of the Christoffel symbols of the
        Fisher information metric.

        For computation purposes, we replace the value of
        (gs.polygamma(1, x) - 1/x) and (gs.polygamma(2,x) + 1/x**2) by an equivalent
        (close bounds) when they become too difficult to compute.

        References
        ----------
        ..[GQ2015] Guo, B. N., Qi, F., Zhao, J. L., & Luo, Q. M. (2015).
            Sharp inequalities for polygamma functions.
            Mathematica Slovaca, 65(1), 103-120.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        jac : array-like, shape=[..., 2, 2, 2, 2]
            Jacobian of the Christoffel symbols.
            :math: 'jac[..., i, j, k, l] = dGamma^i_{jk} / dx_l'
        """
        base_point = gs.to_ndarray(base_point, 2)

        n_points = base_point.shape[0]

        kappa, gamma = base_point[:, 0], base_point[:, 1]

        term_0 = gs.zeros((n_points))
        term_1 = 1 / gamma**2
        term_2 = gs.where(
            gs.polygamma(1, kappa) - 1 / kappa > gs.atol,
            kappa / (gamma**3 * (kappa * gs.polygamma(1, kappa) - 1)),
            kappa**2 / gamma**3,
        )
        term_3 = -1 / (2 * kappa**2)
        term_4 = gs.where(
            gs.polygamma(1, kappa) - 1 / kappa > gs.atol,
            (kappa**2 * gs.polygamma(2, kappa) + 1)
            / (2 * gamma**2 * (kappa * gs.polygamma(1, kappa) - 1) ** 2),
            (kappa**4 * gs.polygamma(2, kappa) + kappa**2) / (2 * gamma**2),
        )
        term_5 = gs.where(
            gs.polygamma(1, kappa) - 1 / kappa > gs.atol,
            (
                kappa**4
                * (
                    gs.polygamma(1, kappa) * gs.polygamma(3, kappa)
                    - gs.polygamma(2, kappa) ** 2
                )
                - kappa**3 * gs.polygamma(3, kappa)
                - 2 * kappa**2 * gs.polygamma(2, kappa)
                - 2 * kappa * gs.polygamma(1, kappa)
                + 1
            )
            / (2 * (kappa**2 * gs.polygamma(1, kappa) - kappa) ** 2),
            0.5
            * (
                kappa**4
                * (
                    gs.polygamma(1, kappa) * gs.polygamma(3, kappa)
                    - gs.polygamma(2, kappa) ** 2
                )
                - kappa**3 * gs.polygamma(3, kappa)
                - 2 * kappa**2 * gs.polygamma(2, kappa)
                - 2 * kappa * gs.polygamma(1, kappa)
                + 1
            ),
        )

        jac = gs.array(
            [
                [
                    [[term_5, term_0], [term_0, term_0]],
                    [[term_0, term_0], [term_4, term_2]],
                ],
                [
                    [[term_0, term_0], [term_3, term_0]],
                    [[term_3, term_0], [term_0, term_1]],
                ],
            ]
        )

        if n_points > 1:
            jac = gs.transpose(jac, [4, 0, 1, 2, 3])

        return gs.squeeze(jac)

    def exp(
        self,
        tangent_vec,
        base_point,
        n_steps=N_STEPS,
        step="euler",
        solver="geomstats",
    ):
        """Compute the exponential map.

        Comute the exponential map associated to the Fisher information metric
        by solving the initial value problem associated to the geodesic
        ordinary differential equation (ODE) using the Christoffel symbols.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Base point.
        n_steps : int
            Number of steps for integration.
            Optional, default: 100.
        solver : string, {'geomstats', 'lsoda'}
            Solver to compute the exponential map (self._exp_ivp or Connection.exp)
            Optional, default : "geomstats"
        step : str, {'euler', 'rk4'}
            Numerical scheme to use for integration in the geomstats solver.
            Optional, default: 'euler'.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            End point of the geodesic starting at base_point with
            initial velocity tangent_vec and stopping at time 1.
        """
        if solver == "geomstats":
            return super().exp(
                tangent_vec,
                base_point,
                n_steps,
                step,
            )
        if solver == "lsoda":
            return self._exp_ivp(tangent_vec, base_point, n_steps)

    def log(
        self,
        point,
        base_point,
        n_steps=N_STEPS,
        init=None,
        step="euler",
        max_iter=25,
        verbose=False,
        tol=gs.atol,
        method="geodesic_shooting",
    ):
        """Compute the logarithm map.

        Compute logarithm map associated to the Fisher information metric by
        solving the boundary value problem associated to the geodesic ordinary
        differential equation (ODE) using the Christoffel symbols.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.
        base_point : array-like, shape=[..., dim]
            Base point.
        n_steps : int
            Number of steps for integration.
            Optional, default: 100.
        step : str, {'euler', 'rk4'}
            Numerical scheme to use for integration.
            Optional, default: 'euler'.
        max_iter
        verbose
        tol
        method : string, {'geodesic_shooting', 'ode_bvp'}
            Method to compute the logarithm map (self._log_bvp or Connection.log)
            Optional, default : 'geodesic_shooting'


        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Initial velocity of the geodesic starting at base_point and
            reaching point at time 1.
        """
        try:
            if method == "geodesic_shooting":
                return super().log(
                    point, base_point, n_steps, step, max_iter, verbose, tol
                )
            if method == "ode_bvp":
                return self._log_bvp(point, base_point, n_steps, jacobian=True)
        except RuntimeWarning:
            print(
                "The points at which to compute the log might be too far away in "
                "terms of Riemmanian distance."
                " Trying with a higher n_steps may solve this issue."
            )
        except UserWarning:
            print(
                "The points at which to compute the log might be too far away in "
                "terms of Riemmanian distance."
                " Trying with a higher n_steps may solve this issue."
            )

    def geodesic(
        self,
        initial_point,
        end_point=None,
        initial_tangent_vec=None,
        n_steps=N_STEPS,
        solver="geomstats",
    ):
        """Generate parameterized function for the geodesic curve.

        Geodesic curve defined by either:

        - an initial point and an initial tangent vector,
        - an initial point and an end point.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Point on the manifold, initial point of the geodesic.
        end_point : array-like, shape=[..., dim], optional
            Point on the manifold, end point of the geodesic. If None,
            an initial tangent vector must be given.
        initial_tangent_vec : array-like, shape=[..., dim],
            Tangent vector at base point, the initial speed of the geodesics.
            Optional, default: None.
            If None, an end point must be given and a logarithm is computed.
        solver : string, {'geomstats', 'vp'}
            Solver to compute the log map (self.__geodesic_vp or Connection.geodesic)
            Optional, default : 'geomstats'

        Returns
        -------
        path : callable
            Time parameterized geodesic curve. If a batch of initial
            conditions is passed, the output array's first dimension
            represents time, and the second corresponds to the different
            initial conditions.
        """
        if solver == "geomstats":
            return super().geodesic(
                initial_point, end_point, initial_tangent_vec, n_steps=n_steps
            )
        if solver == "vp":
            return self._geodesic_vp(
                initial_point, end_point, initial_tangent_vec, n_steps=n_steps
            )

    def _geodesic_ivp(self, initial_point, initial_tangent_vec, n_steps=N_STEPS):
        """Solve geodesic initial value problem.

        Compute the parameterized function for the geodesic starting at
        initial_point with initial velocity given by initial_tangent_vec.
        This is acheived by integrating the geodesic equation.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Initial point.

        initial_tangent_vec : array-like, shape=[..., dim]
            Tangent vector at initial point.

        Returns
        -------
        path : function
            Parameterized function for the geodesic curve starting at
            initial_point with velocity initial_tangent_vec.
        """
        initial_point = gs.to_ndarray(initial_point, to_ndim=2)
        initial_tangent_vec = gs.to_ndarray(initial_tangent_vec, to_ndim=2)

        n_initial_points = initial_point.shape[0]
        n_initial_tangent_vecs = initial_tangent_vec.shape[0]
        if n_initial_points > n_initial_tangent_vecs:
            raise ValueError(
                "There cannot be more initial points than initial tangent vectors."
            )
        if n_initial_tangent_vecs > n_initial_points:
            if n_initial_points > 1:
                raise ValueError(
                    "For several initial tangent vectors, "
                    "specify either one or the same number of "
                    "initial points."
                )
            initial_point = gs.tile(initial_point, (n_initial_tangent_vecs, 1))

        def ivp(state, _):
            """Reformat the initial value problem geodesic ODE."""
            position, velocity = state[: self.dim], state[self.dim :]
            state = gs.stack([position, velocity])
            vel, acc = self.geodesic_equation(state, _)
            eq = (vel, acc)
            return gs.hstack(eq)

        def path(t):
            """Generate parameterized function for geodesic curve.

            Parameters
            ----------
            t : array-like, shape=[n_times,]
                Times at which to compute points of the geodesics.

            Returns
            -------
            geodesic : array-like, shape=[..., n_times, dim]
                Values of the geodesic at times t.
            """
            t = gs.to_ndarray(t, to_ndim=1)
            n_times = len(t)
            geod = []

            if n_times < n_steps:
                t_int = gs.linspace(0, 1, n_steps + 1)
                tangent_vecs = gs.einsum("i,...k->...ik", t, initial_tangent_vec)
                for point, vec in zip(initial_point, tangent_vecs):
                    point = gs.tile(point, (n_times, 1))
                    exp = []
                    for pt, vc in zip(point, vec):
                        initial_state = gs.hstack([pt, vc])
                        solution = odeint(ivp, initial_state, t_int, ())
                        shooting_point = solution[-1, : self.dim]
                        exp.append(shooting_point)
                    exp = gs.array(exp) if n_times == 1 else gs.stack(exp)
                    geod.append(exp)
            else:
                t_int = t
                for point, vec in zip(initial_point, initial_tangent_vec):
                    initial_state = gs.hstack([point, vec])
                    solution = odeint(ivp, initial_state, t_int, ())
                    geod.append(solution[:, : self.dim])

            geod = geod[0] if len(initial_point) == 1 else gs.stack(geod)
            return gs.where(geod < gs.atol, gs.atol, geod)

        return path

    def _exp_ivp(self, tangent_vec, base_point, n_steps=N_STEPS):
        """Compute the exponential map.

        Comute the exponential map associated to the Fisher information metric
        by solving the initial value problem associated to the geodesic
        ordinary differential equation (ODE) using the Christoffel symbols.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Base point.
        n_steps : int
            Number of steps for integration.
            Optional, default: 100.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            End point of the geodesic starting at base_point with
            initial velocity tangent_vec and stopping at time 1.
        """
        stop_time = 1.0
        geodesic = self._geodesic_ivp(base_point, tangent_vec, n_steps)
        exp = gs.squeeze(geodesic(stop_time)[..., 0, :])

        return exp

    def _approx_geodesic_bvp(
        self,
        initial_point,
        end_point,
        degree=5,
        method="BFGS",
        n_times=200,
        jac_on=True,
    ):
        """Solve approximation of the geodesic boundary value problem.

        The space of solutions is restricted to curves whose coordinates are
        polynomial functions of time. The boundary value problem is solved by
        minimizing the energy among all such curves starting from initial_point
        and ending at end_point, i.e. curves t -> (x_1(t),...,x_n(t)) where x_i
        are polynomial functions of time t, such that (x_1(0),..., x_n(0)) is
        initial_point and (x_1(1),..., x_n(1)) is end_point. The parameterized
        curve is computed at n_times discrete times.

        Parameters
        ----------
        initial_point : array-like, shape=(dim,)
            Starting point of the geodesic.
        end_point : array-like, shape=(dim,)
            End point of the geodesic.
        degree : int
            Degree of the coordinates' polynomial functions of time.
        method : str
            Minimization method to use in scipy.optimize.minimize.
        n_times : int
            Number of sample times.
        jac_on : bool
            If jac_on=True, use the Jacobian of the energy cost function in
            scipy.optimize.minimize.

        Returns
        -------
        dist : float
            Length of the polynomial approximation of the geodesic.
        curve : array-like, shape=(n_times, dim)
            Polynomial approximation of the geodesic.
        velocity : array-like, shape=(n_times, dim)
            Velocity of the polynomial approximation of the geodesic.
        """

        def cost_fun(param):
            """Compute the energy of the polynomial curve defined by param.

            Parameters
            ----------
            param : array-like, shape=(degree - 1, dim)
                Parameters of the curve coordinates' polynomial functions of time.

            Returns
            -------
            energy : float
                Energy of the polynomial approximation of the geodesic.
            length : float
                Length of the polynomial approximation of the geodesic.
            curve : array-like, shape=(n_times, dim)
                Polynomial approximation of the geodesic.
            velocity : array-like, shape=(n_times, dim)
                Velocity of the polynomial approximation of the geodesic.
            """
            last_coef = end_point - initial_point - gs.sum(param, axis=0)
            coef = gs.vstack((initial_point, param, last_coef))

            t = gs.linspace(0.0, 1.0, n_times)
            t_curve = [t**i for i in range(degree + 1)]
            t_curve = gs.stack(t_curve)
            curve = gs.einsum("ij,ik->kj", coef, t_curve)

            t_velocity = [i * t ** (i - 1) for i in range(1, degree + 1)]
            t_velocity = gs.stack(t_velocity)
            velocity = gs.einsum("ij,ik->kj", coef[1:], t_velocity)

            if curve.min() < 0:
                return np.inf, np.inf, curve, np.nan

            velocity_sqnorm = self.squared_norm(vector=velocity, base_point=curve)
            length = gs.sum(velocity_sqnorm ** (1 / 2)) / n_times
            energy = gs.sum(velocity_sqnorm) / n_times
            return energy, length, curve, velocity

        def cost_jacobian(param):
            """Compute the jacobian of the cost function at polynomial curve.

            Parameters
            ----------
            param : array-like, shape=(degree - 1, dim)
                Parameters of the curve coordinates' polynomial functions of time.

            Returns
            -------
            jac : array-like, shape=(dim * (degree - 1),)
                Jacobian of the cost function at polynomial curve.
            """
            last_coef = end_point - initial_point - gs.sum(param, axis=0)
            coef = gs.vstack((initial_point, param, last_coef))

            t = gs.linspace(0.0, 1.0, n_times)
            t_position = [t**i for i in range(degree + 1)]
            t_position = gs.stack(t_position)
            position = gs.einsum("ij,ik->kj", coef, t_position)

            t_velocity = [i * t ** (i - 1) for i in range(1, degree + 1)]
            t_velocity = gs.stack(t_velocity)
            velocity = gs.einsum("ij,ik->kj", coef[1:], t_velocity)

            kappa, gamma = position[:, 0], position[:, 1]
            kappa_dot, gamma_dot = velocity[:, 0], velocity[:, 1]

            jac_kappa_0 = (
                (gs.polygamma(2, kappa) + 1 / kappa**2) * kappa_dot
                + gamma_dot**2 / gamma
            ) * t_position[1:-1]
            jac_kappa_1 = (2 * gs.polygamma(1, kappa) * kappa_dot) * t_velocity[:-1]

            jac_kappa = jac_kappa_0 + jac_kappa_1

            jac_gamma_0 = (-kappa * gamma_dot**2 / gamma**2) * t_position[1:-1]
            jac_gamma_1 = (2 * kappa * gamma_dot / gamma) * t_velocity[:-1]

            jac_gamma = jac_gamma_0 + jac_gamma_1

            jac = gs.vstack([jac_kappa, jac_gamma])

            cost_jac = gs.sum(jac, axis=1)
            return cost_jac

        def f2minimize(x):
            """Compute function to minimize."""
            param = gs.transpose(x.reshape((dim, degree - 1)))
            res = cost_fun(param)
            return res[0]

        def jacobian(x):
            """Compute jacobian of the function to minimize."""
            param = gs.transpose(x.reshape((dim, degree - 1)))
            return cost_jacobian(param)

        dim = initial_point.shape[0]
        x0 = gs.ones(dim * (degree - 1))
        jac = jacobian if jac_on else None
        sol = minimize(f2minimize, x0, method=method, jac=jac)
        opt_param = sol.x.reshape((dim, degree - 1)).T
        _, dist, curve, velocity = cost_fun(opt_param)

        return dist, curve, velocity

    def _geodesic_bvp(
        self,
        initial_point,
        end_point,
        n_steps=N_STEPS,
        jacobian=True,
    ):
        """Solve geodesic boundary problem.

        Compute the parameterized function for the geodesic starting at
        initial_point and ending at end_point. This is acheived by integrating
        the geodesic equation.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Initial point.
        end_point : array-like, shape=[..., dim]
            End point.
        jacobian : boolean.
            If True, the explicit value of the jacobian is used to solve
            the geodesic boundary value problem.
            Optional, default: True.

        Returns
        -------
        path : function
            Parameterized function for the geodesic curve starting at
            initial_point and ending at end_point.
        """
        initial_point = gs.to_ndarray(initial_point, to_ndim=2)
        end_point = gs.to_ndarray(end_point, to_ndim=2)
        n_initial_points = initial_point.shape[0]
        n_end_points = end_point.shape[0]
        if n_initial_points > n_end_points:
            if n_end_points > 1:
                raise ValueError(
                    "For several initial points, specify either"
                    "one or the same number of end points."
                )
            end_point = gs.tile(end_point, (n_initial_points, 1))
        elif n_end_points > n_initial_points:
            if n_initial_points > 1:
                raise ValueError(
                    "For several end points, specify either "
                    "one or the same number of initial points."
                )
            initial_point = gs.tile(initial_point, (n_end_points, 1))

        def bvp(_, state):
            """Reformat the boundary value problem geodesic ODE.

            Parameters
            ----------
            state :  array-like, shape[2 * dim,]
                Vector of the state variables: position and speed.
            _ :  unused
                Any (time).
            """
            position, velocity = state[: self.dim].T, state[self.dim :].T
            state = gs.stack([position, velocity])
            vel, acc = self.geodesic_equation(state, _)
            eq = (vel, acc)
            return gs.transpose(gs.hstack(eq))

        def boundary_cond(state_0, state_1, point_0, point_1):
            """Boundary condition for the geodesic ODE."""
            return gs.hstack(
                (state_0[: self.dim] - point_0, state_1[: self.dim] - point_1)
            )

        def jac(_, state):
            """Jacobian of bvp function.

            Parameters
            ----------
            state :  array-like, shape=[2*dim, ...]
                Vector of the state variables (position and speed)
            _ :  unused
                Any (time).

            Returns
            -------
            jac : array-like, shape=[dim, dim, ...]
            """
            n_times = state.shape[1]
            position, velocity = state[: self.dim], state[self.dim :]

            dgamma = self.jacobian_christoffels(gs.transpose(position))

            df_dposition = -gs.einsum(
                "j...,...ijkl,k...->il...", velocity, dgamma, velocity
            )

            gamma = self.christoffels(gs.transpose(position))
            df_dvelocity = -2 * gs.einsum("...ijk,k...->ij...", gamma, velocity)

            jac_nw = gs.squeeze((gs.zeros((self.dim, self.dim, n_times))))
            jac_ne = gs.squeeze(
                gs.transpose(gs.tile(gs.eye(self.dim), (n_times, 1, 1)))
            )
            jac_sw = df_dposition
            jac_se = df_dvelocity
            jac = gs.concatenate(
                (
                    gs.concatenate((jac_nw, jac_ne), axis=1),
                    gs.concatenate((jac_sw, jac_se), axis=1),
                ),
                axis=0,
            )

            return jac

        def path(t):
            """Generate parameterized function for geodesic curve.

            Parameters
            ----------
            t : array-like, shape=[n_times,]
                Times at which to compute points of the geodesics.

            Returns
            -------
            geodesic : array-like, shape=[..., n_times, dim]
                Values of the geodesic at times t.
            """
            t = gs.to_ndarray(t, to_ndim=1)
            geod = []

            def initialize(point_0, point_1, init="polynomial"):
                """Initialize the solution of the boundary value problem."""
                if init == "polynomial":
                    _, curve, velocity = self._approx_geodesic_bvp(
                        point_0, point_1, n_times=n_steps
                    )
                    return gs.vstack((curve.T, velocity.T))

                lin_init = gs.zeros([2 * self.dim, n_steps])
                lin_init[: self.dim, :] = gs.transpose(
                    gs.linspace(point_0, point_1, n_steps)
                )
                lin_init[self.dim :, :-1] = n_steps * (
                    lin_init[: self.dim, 1:] - lin_init[: self.dim, :-1]
                )
                lin_init[self.dim :, -1] = lin_init[self.dim :, -2]
                return lin_init

            t_int = gs.linspace(0.0, 1.0, n_steps)
            fun_jac = jac if jacobian else None

            for ip, ep in zip(initial_point, end_point):

                def bc(y0, y1, ip=ip, ep=ep):
                    return boundary_cond(y0, y1, ip, ep)

                solution = solve_bvp(
                    bvp, bc, t_int, initialize(ip, ep), fun_jac=fun_jac, max_nodes=10000
                )
                if solution.status == 1:
                    logging.warning(
                        "The maximum number of mesh nodes for solving the "
                        "geodesic boundary value problem is exceeded."
                    )
                solution_at_t = solution.sol(t)
                geodesic = solution_at_t[: self.dim, :]
                geod.append(gs.squeeze(gs.transpose(geodesic)))

            geod = geod[0] if len(initial_point) == 1 else gs.stack(geod)
            return gs.where(geod < gs.atol, gs.atol, geod)

        return path

    def _log_bvp(
        self, point, base_point, n_steps=N_STEPS, jacobian=True, init="polynomial"
    ):
        """Compute the logarithm map.

        Compute logarithm map associated to the Fisher information metric by
        solving the boundary value problem associated to the geodesic ordinary
        differential equation (ODE) using the Christoffel symbols.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.
        base_point : array-like, shape=[..., dim]
            Base po int.
        n_steps : int
            Number of steps for integration.
            Optional, default: 100.
        jacobian : boolean.
            If True, the explicit value of the jacobian is used to solve
            the geodesic boundary value problem.
            Optional, default: True.
        init : str, {'linear', 'polynomial}
            Initialization used to solve the geodesic boundary value problem.
            If 'linear', use the Euclidean straight line as initial guess.
            If 'polynomial', use a curve with coordinates that are polynomial
            functions of time.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Initial velocity of the geodesic starting at base_point and
            reaching point at time 1.
        """
        t = gs.linspace(0.0, 1.0, n_steps)
        geodesic = self._geodesic_bvp(
            initial_point=base_point,
            end_point=point,
            jacobian=jacobian,
        )
        geodesic_at_t = geodesic(t)
        log = n_steps * (geodesic_at_t[..., 1, :] - geodesic_at_t[..., 0, :])

        return gs.squeeze(gs.stack(log))

    def _geodesic_vp(
        self,
        initial_point,
        end_point=None,
        initial_tangent_vec=None,
        n_steps=N_STEPS,
        jacobian=True,
    ):
        """Generate parameterized function for the geodesic curve.

        Geodesic curve defined by either:

        - an initial point and an initial tangent vector,
        - an initial point and an end point.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Point on the manifold, initial point of the geodesic.
        end_point : array-like, shape=[..., dim], optional
            Point on the manifold, end point of the geodesic. If None,
            an initial tangent vector must be given.
        initial_tangent_vec : array-like, shape=[..., dim],
            Tangent vector at base point, the initial speed of the geodesics.
            Optional, default: None.
            If None, an end point must be given and a logarithm is computed.
        jacobian : boolean.
            If True, the explicit value of the jacobian is used to solve
            the geodesic boundary value problem.
            Optional, default: True.

        Returns
        -------
        path : callable
            Time parameterized geodesic curve. If a batch of initial
            conditions is passed, the output array's first dimension
            represents time, and the second corresponds to the different
            initial conditions.
        """
        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end point or an initial tangent "
                "vector to define the geodesic."
            )
        if end_point is not None:
            if initial_tangent_vec is not None:
                raise ValueError(
                    "Cannot specify both an end point " "and an initial tangent vector."
                )
            path = self._geodesic_bvp(
                initial_point, end_point, n_steps, jacobian=jacobian
            )

        if initial_tangent_vec is not None:
            path = self._geodesic_ivp(initial_point, initial_tangent_vec, n_steps)

        return path
