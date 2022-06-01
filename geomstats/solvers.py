import logging
from abc import ABCMeta, abstractmethod

# TODO: numpy needs to be removed
import numpy as np
from scipy.integrate import odeint, solve_bvp
from scipy.optimize import minimize

import geomstats.backend as gs
from geomstats.integrator import integrate


class LogSolver(metaclass=ABCMeta):
    @abstractmethod
    def solve(self, metric, point, base_point):
        pass


class GeomstatsLogSolver(LogSolver):
    def __init__(self, method="L-BFGS-B", max_iter=25, verbose=False, atol=gs.atol):
        super().__init__()

        self.method = method
        self.max_iter = max_iter
        self.verbose = verbose
        self.atol = atol

        self.res_ = None

    def solve(self, metric, point, base_point):
        """Compute logarithm map associated to the affine connection.

        Solve the boundary value problem associated to the geodesic equation
        using the Christoffel symbols and conjugate gradient descent.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point on the manifold.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_steps : int
            Number of discrete time steps to take in the integration.
            Optional, default: N_STEPS.
        step : str, {'euler', 'rk4'}
            Numerical scheme to use for integration.
            Optional, default: 'euler'.
        max_iter
        verbose
        tol

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        """
        max_shape = point.shape
        if len(point.shape) <= len(base_point.shape):
            max_shape = base_point.shape

        def objective(velocity):
            """Define the objective function."""
            velocity = gs.array(velocity, dtype=base_point.dtype)
            velocity = gs.reshape(velocity, max_shape)
            delta = metric.exp(velocity, base_point) - point
            return gs.sum(delta**2)

        objective_with_grad = gs.autodiff.value_and_grad(objective, to_numpy=True)

        tangent_vec = gs.flatten(gs.random.rand(*max_shape))

        res = minimize(
            objective_with_grad,
            tangent_vec,
            method=self.method,
            jac=True,
            options={"disp": self.verbose, "maxiter": self.max_iter},
            tol=self.atol,
        )
        self.res_ = res

        tangent_vec = gs.array(res.x, dtype=base_point.dtype)
        tangent_vec = gs.reshape(tangent_vec, max_shape)
        return tangent_vec


class ExpSolver(metaclass=ABCMeta):
    @abstractmethod
    def solve(self, metric, tangent_vec, base_point):
        pass


class GeomstatsExpSolver(ExpSolver):
    def __init__(self, n_steps=10, step="euler"):
        super().__init__()
        self.n_steps = n_steps
        self.step = step

        self.flow_ = None

    def solve(self, metric, tangent_vec, base_point):
        """Exponential map associated to the affine connection.

        Exponential map at base_point of tangent_vec computed by integration
        of the geodesic equation (initial value problem), using the
        christoffel symbols.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_steps : int
            Number of discrete time steps to take in the integration.
            Optional, default: N_STEPS.
        step : str, {'euler', 'rk4'}
            The numerical scheme to use for integration.
            Optional, default: 'euler'.
        point_type : str, {'vector', 'matrix'}
            Type of representation used for points.
            Optional, default: None.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point on the manifold.
        """
        base_point = gs.broadcast_to(base_point, tangent_vec.shape)

        initial_state = gs.stack([base_point, tangent_vec])

        flow = integrate(
            metric.geodesic_equation,
            initial_state,
            n_steps=self.n_steps,
            step=self.step,
        )
        self.flow_ = flow

        exp = flow[-1][0]
        return exp


class GeodesicSolver(metaclass=ABCMeta):
    @abstractmethod
    def geodesic(self, metric, initial_point, end_point=None, initial_tangent_vec=None):
        pass


class GeomstatsGeodesicSolver(GeodesicSolver):
    def geodesic(
        self,
        metric,
        initial_point,
        end_point=None,
        initial_tangent_vec=None,
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

        Returns
        -------
        path : callable
            Time parameterized geodesic curve. If a batch of initial
            conditions is passed, the output array's first dimension
            represents the different initial conditions, and the second
            corresponds to time.
        """
        point_type = metric.default_point_type

        if end_point is not None:
            initial_tangent_vec = metric.log(point=end_point, base_point=initial_point)

        if point_type == "vector":
            initial_point = gs.to_ndarray(initial_point, to_ndim=2)
            initial_tangent_vec = gs.to_ndarray(initial_tangent_vec, to_ndim=2)

        else:
            initial_point = gs.to_ndarray(initial_point, to_ndim=3)
            initial_tangent_vec = gs.to_ndarray(initial_tangent_vec, to_ndim=3)
        n_initial_conditions = initial_tangent_vec.shape[0]

        if n_initial_conditions > 1 and len(initial_point) == 1:
            initial_point = gs.stack([initial_point[0]] * n_initial_conditions)

        def path(t):
            """Generate parameterized function for geodesic curve.

            Parameters
            ----------
            t : array-like, shape=[n_points,]
                Times at which to compute points of the geodesics.
            """
            t = gs.array(t)
            t = gs.cast(t, initial_tangent_vec.dtype)
            t = gs.to_ndarray(t, to_ndim=1)
            if point_type == "vector":
                tangent_vecs = gs.einsum("i,...k->...ik", t, initial_tangent_vec)
            else:
                tangent_vecs = gs.einsum("i,...kl->...ikl", t, initial_tangent_vec)

            points_at_time_t = [
                metric.exp(tv, pt) for tv, pt in zip(tangent_vecs, initial_point)
            ]
            points_at_time_t = gs.stack(points_at_time_t, axis=0)

            return (
                points_at_time_t[0] if n_initial_conditions == 1 else points_at_time_t
            )

        return path


class GammaGeodesicVPSolver(GeodesicSolver):
    # TODO: make it generic

    def __init__(self, jacobian=True, n_steps=10):
        super().__init__()
        self.jacobian = jacobian
        self.n_steps = n_steps

    class ExpIVPSolver(ExpSolver):
        def solve(self, metric, tangent_vec, base_point):
            geodesic = metric.geodesic(tangent_vec, base_point)
            return gs.squeeze(geodesic(1.0)[..., 0, :])

    class LogBVPSolver(LogSolver):
        def __init__(self, n_steps=10):
            self.n_steps = n_steps

        def solve(self, metric, point, base_point):
            t = gs.linspace(0.0, 1.0, self.n_steps)
            geodesic = metric.geodesic(initial_point=base_point, end_point=point)
            geodesic_at_t = geodesic(t)
            log = self.n_steps * (geodesic_at_t[..., 1, :] - geodesic_at_t[..., 0, :])

            return gs.squeeze(gs.stack(log))

    def get_log_exp_solvers(self):
        return self.LogBVPSolver(n_steps=self.n_steps), self.ExpIVPSolver()

    def _geodesic_bvp(self, metric, initial_point, end_point):
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
        # TODO: to_ndim
        initial_point = gs.to_ndarray(initial_point, to_ndim=2)
        end_point = gs.to_ndarray(end_point, to_ndim=2)
        n_initial_points = initial_point.shape[0]
        n_end_points = end_point.shape[0]

        # TODO: broadcast
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
            # TODO: metric.dim
            position, velocity = state[: metric.dim].T, state[metric.dim :].T
            state = gs.stack([position, velocity])
            vel, acc = metric.geodesic_equation(state, _)
            eq = (vel, acc)
            return gs.transpose(gs.hstack(eq))

        def boundary_cond(state_0, state_1, point_0, point_1):
            """Boundary condition for the geodesic ODE."""
            return gs.hstack(
                (state_0[: metric.dim] - point_0, state_1[: metric.dim] - point_1)
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
            position, velocity = state[: metric.dim], state[metric.dim :]

            dgamma = metric.jacobian_christoffels(gs.transpose(position))

            df_dposition = -gs.einsum(
                "j...,...ijkl,k...->il...", velocity, dgamma, velocity
            )

            gamma = metric.christoffels(gs.transpose(position))
            df_dvelocity = -2 * gs.einsum("...ijk,k...->ij...", gamma, velocity)

            jac_nw = gs.squeeze((gs.zeros((metric.dim, metric.dim, n_times))))
            jac_ne = gs.squeeze(
                gs.transpose(gs.tile(gs.eye(metric.dim), (n_times, 1, 1)))
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
                        metric, point_0, point_1, n_times=self.n_steps
                    )
                    return gs.vstack((curve.T, velocity.T))

                lin_init = gs.zeros([2 * metric.dim, self.n_steps])
                lin_init[: metric.dim, :] = gs.transpose(
                    gs.linspace(point_0, point_1, self.n_steps)
                )
                lin_init[metric.dim :, :-1] = self.n_steps * (
                    lin_init[: metric.dim, 1:] - lin_init[: metric.dim, :-1]
                )
                lin_init[metric.dim :, -1] = lin_init[metric.dim :, -2]
                return lin_init

            t_int = gs.linspace(0.0, 1.0, self.n_steps)
            fun_jac = jac if self.jacobian else None

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
                geodesic = solution_at_t[: metric.dim, :]
                geod.append(gs.squeeze(gs.transpose(geodesic)))

            geod = geod[0] if len(initial_point) == 1 else gs.stack(geod)
            return gs.where(geod < gs.atol, gs.atol, geod)

        return path

    def _geodesic_ivp(self, metric, initial_point, initial_tangent_vec):
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
            position, velocity = state[: metric.dim], state[metric.dim :]
            state = gs.stack([position, velocity])
            vel, acc = metric.geodesic_equation(state, _)
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

            if n_times < self.n_steps:
                t_int = gs.linspace(0, 1, self.n_steps + 1)
                tangent_vecs = gs.einsum("i,...k->...ik", t, initial_tangent_vec)
                for point, vec in zip(initial_point, tangent_vecs):
                    point = gs.tile(point, (n_times, 1))
                    exp = []
                    for pt, vc in zip(point, vec):
                        initial_state = gs.hstack([pt, vc])
                        solution = odeint(ivp, initial_state, t_int, ())
                        shooting_point = solution[-1, : metric.dim]
                        exp.append(shooting_point)
                    exp = gs.array(exp) if n_times == 1 else gs.stack(exp)
                    geod.append(exp)
            else:
                t_int = t
                for point, vec in zip(initial_point, initial_tangent_vec):
                    initial_state = gs.hstack([point, vec])
                    solution = odeint(ivp, initial_state, t_int, ())
                    geod.append(solution[:, : metric.dim])

            geod = geod[0] if len(initial_point) == 1 else gs.stack(geod)
            return gs.where(geod < gs.atol, gs.atol, geod)

        return path

    def geodesic(
        self,
        metric,
        initial_point,
        end_point=None,
        initial_tangent_vec=None,
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
            path = self._geodesic_bvp(metric, initial_point, end_point)

        if initial_tangent_vec is not None:
            path = self._geodesic_ivp(metric, initial_point, initial_tangent_vec)

        return path

    def _approx_geodesic_bvp(
        self,
        metric,
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
        # TODO: can be coded better

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

            velocity_sqnorm = metric.squared_norm(vector=velocity, base_point=curve)
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
