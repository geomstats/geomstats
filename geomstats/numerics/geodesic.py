"""Geodesic solvers implementation."""

from abc import ABC, abstractmethod

import geomstats.backend as gs
from geomstats.numerics.bvp import ScipySolveBVP
from geomstats.numerics.ivp import GSIVPIntegrator
from geomstats.numerics.optimizers import ScipyMinimize


class ExpSolver(ABC):
    """Abstract class for geodesic initial value problem solvers.

    Parameters
    ----------
    solves_ivp : bool
        Informs if solver is able to solve for geodesic at different t.
    """

    def __init__(self, solves_ivp=False):
        self.solves_ivp = solves_ivp

    @abstractmethod
    def exp(self, space, tangent_vec, base_point):
        """Exponential map.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        end_point : array-like, shape=[..., dim]
            Point on the manifold.
        """

    def geodesic_ivp(self, space, tangent_vec, base_point):
        """Geodesic curve for initial value problem.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        path : callable
            Time parametrized geodesic curve. `f(t)`.
        """
        raise NotImplementedError("Can't solve for geodesic at different t.")


class ExpODESolver(ExpSolver):
    """Geodesic initial value problem solver.

    Parameters
    ----------
    integrator : ODEIVPSolver
        Instance of ODEIVP integrator.
    """

    def __init__(self, integrator=None):
        super().__init__()

        if integrator is None:
            integrator = GSIVPIntegrator()

        self._integrator = None
        self.integrator = integrator

    @property
    def integrator(self):
        """An instance of ODEIVPSolver."""
        return self._integrator

    @integrator.setter
    def integrator(self, integrator):
        """Set integrator."""
        self.solves_ivp = integrator.tchosen
        self._integrator = integrator

    def _solve(self, space, tangent_vec, base_point, t_eval=None):
        if base_point.ndim != tangent_vec.ndim:
            base_point = gs.broadcast_to(base_point, tangent_vec.shape)

        state_axis = -(space.point_ndim + 1)
        initial_state = gs.stack([base_point, tangent_vec], axis=state_axis)

        force = space.metric.geodesic_equation
        if t_eval is None:
            return self.integrator.integrate(force, initial_state)

        return self.integrator.integrate_t(force, initial_state, t_eval)

    def exp(self, space, tangent_vec, base_point):
        """Exponential map.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.

        Returns
        -------
        end_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        """
        result = self._solve(space, tangent_vec, base_point)
        return self._simplify_exp_result(result, space)

    def geodesic_ivp(self, space, tangent_vec, base_point):
        """Geodesic curve for initial value problem.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.

        Returns
        -------
        path : callable
            Time parametrized geodesic curve. `f(t)`.
        """
        if not self.solves_ivp:
            raise NotImplementedError(
                "Can't solve for geodesic at different t with this integrator."
            )

        base_point = gs.broadcast_to(base_point, tangent_vec.shape)

        def path(t):
            """Time parametrized geodesic curve.

            Parameters
            ----------
            t : float or array-like, shape=[n_times,]

            Returns
            -------
            geodesic_points : array-like, shape=[..., n_times, *space.shape]
                Geodesic points evaluated at t.
            """
            if not gs.is_array(t):
                t = gs.array([t])

            if gs.ndim(t) == 0:
                t = gs.expand_dims(t, axis=0)

            result = self._solve(space, tangent_vec, base_point, t_eval=t)
            return self._simplify_result_t(result, space)

        return path

    def _simplify_exp_result(self, result, space):
        y = result.get_last_y()
        slc = tuple([slice(None)] * space.point_ndim)
        return y[..., 0, *slc]

    def _simplify_result_t(self, result, space):
        # assumes several t
        y = result.y
        slc = tuple([slice(None)] * space.point_ndim)
        return y[..., :, 0, *slc]


class LogSolver(ABC):
    """Abstract class for geodesic boundary value problem solvers.

    Parameters
    ----------
    solves_bvp : bool
        Informs if solver is able to solve for geodesic at different t.
    """

    def __init__(self, solves_bvp=False):
        self.solves_bvp = solves_bvp

    @abstractmethod
    def log(self, space, point, base_point):
        """Logarithm map.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        end_point : array-like, shape=[..., dim]
            Point on the manifold.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        """

    def geodesic_bvp(self, space, point, base_point):
        """Geodesic curve for boundary value problem.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        end_point : array-like, shape=[..., dim]
            Point on the manifold.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        path : callable
            Time parametrized geodesic curve. `f(t)`.
        """
        raise NotImplementedError("Can't solve for geodesic at different t.")


class _LogBatchMixins:
    """Provides method to compute log for multiples point."""

    @abstractmethod
    def _log_single(self, space, point, base_point):
        """Logarithm map.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        end_point : array-like, shape=[dim]
            Point on the manifold.
        base_point : array-like, shape=[dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[dim]
            Tangent vector at the base point.
        """

    def log(self, space, point, base_point):
        """Logarithm map.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        end_point : array-like, shape=[..., dim]
            Point on the manifold.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        """
        # assumes inability to properly vectorize
        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        is_batch = point.ndim > space.point_ndim
        if not is_batch:
            return self._log_single(space, point, base_point)

        return gs.stack(
            [
                self._log_single(space, point_, base_point_)
                for point_, base_point_ in zip(point, base_point)
            ]
        )


class LogShootingSolver:
    """Geodesic boundary value problem solver using shooting.

    Parameters
    ----------
    optimizer : ScipyMinimize
        Instance of ScipyMinimize.
    initialization : callable
        Function to provide initial solution. `f(space, point, base_point)`.
        Defaults to linear initialization.
    flatten : bool
        If True, the optimization problem is solved together for all the points.
    """

    def __new__(cls, optimizer=None, initialization=None, flatten=True):
        """Instantiate a log shooting solver."""
        if flatten:
            return _LogShootingSolverFlatten(
                optimizer=optimizer,
                initialization=initialization,
            )

        return _LogShootingSolverUnflatten(
            optimizer=optimizer,
            initialization=initialization,
        )


class _LogShootingSolverFlatten(LogSolver):
    def __init__(self, optimizer=None, initialization=None):
        super().__init__(solves_bvp=False)

        if optimizer is None:
            optimizer = ScipyMinimize(jac="autodiff")

        if initialization is None:
            initialization = self._default_initialization

        self.optimizer = optimizer
        self.initialization = initialization

    def _default_initialization(self, space, point, base_point):
        return gs.flatten(point - base_point)

    def _objective(self, velocity, space, point, base_point):
        velocity = gs.reshape(velocity, base_point.shape)
        delta = space.metric.exp(velocity, base_point) - point
        return gs.sum(delta**2)

    def log(self, space, point, base_point):
        """Logarithm map.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        end_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at the base point.
        """
        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        objective = lambda velocity: self._objective(velocity, space, point, base_point)
        init_tangent_vec = self.initialization(space, point, base_point)

        res = self.optimizer.minimize(objective, init_tangent_vec)

        return gs.reshape(res.x, base_point.shape)


class _LogShootingSolverUnflatten(_LogBatchMixins, LogSolver):
    def __init__(self, optimizer=None, initialization=None):
        super().__init__(solves_bvp=False)
        if optimizer is None:
            optimizer = ScipyMinimize(jac="autodiff")

        if initialization is None:
            initialization = self._default_initialization

        self.optimizer = optimizer
        self.initialization = initialization

    def _default_initialization(self, space, point, base_point):
        return point - base_point

    def _objective(self, velocity, space, point, base_point):
        if space.point_ndim > 1:
            velocity = gs.reshape(velocity, space.shape)

        delta = space.metric.exp(velocity, base_point) - point
        return gs.sum(delta**2)

    def _log_single(self, space, point, base_point):
        """Logarithm map.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        end_point : array-like, shape=[*space.shape]
            Point on the manifold.
        base_point : array-like, shape=[*space.shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[*space.shape]
            Tangent vector at the base point.
        """
        objective = lambda velocity: self._objective(velocity, space, point, base_point)
        init_tangent_vec = self.initialization(space, point, base_point)

        res = self.optimizer.minimize(objective, gs.flatten(init_tangent_vec))

        if space.point_ndim > 1:
            return gs.reshape(res.x, space.shape)

        return res.x


class LogODESolver(_LogBatchMixins, LogSolver):
    """Geodesic boundary value problem using an ODE solver.

    Parameters
    ----------
    n_nodes : Number of mesh nodes.
    integrator : ScipySolveBVP
        Instance of ScipySolveBVP.
    initialization : callable
        Function to provide initial solution. `f(space, point, base_point)`.
        Defaults to linear initialization.
    """

    def __init__(self, n_nodes=10, integrator=None, initialization=None, use_jac=True):
        super().__init__(solves_bvp=True)

        if integrator is None:
            integrator = ScipySolveBVP()

        if initialization is None:
            initialization = self._default_initialization

        self.n_nodes = n_nodes
        self.integrator = integrator
        self.initialization = initialization
        self.use_jac = use_jac

        self.grid = self._create_grid()

    def _create_grid(self):
        return gs.linspace(0.0, 1.0, num=self.n_nodes)

    def _default_initialization(self, space, point, base_point):
        if point.ndim == 1:
            point_0, point_1 = base_point, point
        else:
            point_0 = gs.flatten(base_point)
            point_1 = gs.flatten(point)

        pos_init = gs.transpose(gs.linspace(point_0, point_1, self.n_nodes))

        vel_init = self.n_nodes * (pos_init[:, 1:] - pos_init[:, :-1])
        vel_init = gs.hstack([vel_init, vel_init[:, [-2]]])

        return gs.vstack([pos_init, vel_init])

    def _boundary_condition(self, state_0, state_1, space, point_0, point_1):
        pos_0 = state_0[: point_0.shape[0]]
        pos_1 = state_1[: point_1.shape[0]]
        return gs.hstack((pos_0 - point_0, pos_1 - point_1))

    def _bvp(self, _, raveled_state, space):
        """Boundary value problem.

        Parameters
        ----------
        _ : float
            Unused.
        raveled_state : array-like, shape=[2*dim, n_nodes]
            Vector of state variables (position and speed).

        Returns
        -------
        sol : array-like, shape=[2*dim, n_nodes]
        """
        state = gs.moveaxis(
            gs.reshape(raveled_state, (2,) + space.shape + (-1,)), -1, 0
        )
        new_state = space.metric.geodesic_equation(state, _)
        return gs.moveaxis(gs.reshape(new_state, (-1, raveled_state.shape[0])), -2, -1)

    def _jacobian(self, _, raveled_state, space):
        """Jacobian of boundary value problem.

        Parameters
        ----------
        _ : float
            Unused.
        raveled_state : array-like, shape=[2*dim, n_nodes]
            Vector of state variables (position and speed).

        Returns
        -------
        jac : array-like, shape=[dim, dim, n_nodes]
        """
        dim = space.dim
        n_nodes = raveled_state.shape[-1]
        position, velocity = raveled_state[:dim], raveled_state[dim:]

        dgamma = space.metric.jacobian_christoffels(gs.transpose(position))

        df_dposition = -gs.einsum(
            "j...,...ijkl,k...->il...", velocity, dgamma, velocity
        )

        gamma = space.metric.christoffels(gs.transpose(position))
        df_dvelocity = -2 * gs.einsum("...ijk,k...->ij...", gamma, velocity)

        jac_nw = gs.zeros((dim, dim, raveled_state.shape[1]))
        jac_ne = gs.squeeze(gs.transpose(gs.tile(gs.eye(dim), (n_nodes, 1, 1))))
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

    def _solve(self, space, point, base_point):
        bvp = lambda t, state: self._bvp(t, state, space)
        bc = lambda state_0, state_1: self._boundary_condition(
            state_0, state_1, space, gs.flatten(base_point), gs.flatten(point)
        )

        jacobian = None
        if self.use_jac:
            jacobian = lambda t, state: self._jacobian(t, state, space=space)

        y = self.initialization(space, point, base_point)

        return self.integrator.integrate(bvp, bc, self.grid, y, fun_jac=jacobian)

    def _log_single(self, space, point, base_point):
        res = self._solve(space, point, base_point)
        return self._simplify_log_result(res, space)

    def geodesic_bvp(self, space, point, base_point):
        """Geodesic curve for boundary value problem.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        end_point : array-like, shape=[..., dim]
            Point on the manifold.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        path : callable
            Time parametrized geodesic curve. `f(t)`. 0 <= t <= 1.
        """
        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        is_batch = point.ndim > space.point_ndim
        if not is_batch:
            result = self._solve(space, point, base_point)
        else:
            results = [
                self._solve(space, point_, base_point_)
                for point_, base_point_ in zip(point, base_point)
            ]

        def path(t):
            """Time parametrized geodesic curve.

            Parameters
            ----------
            t : float or array-like, shape=[n_times,]

            Returns
            -------
            geodesic_points : array-like, shape=[..., n_times, dim]
                Geodesic points evaluated at t.
            """
            if not gs.is_array(t):
                t = gs.array([t])

            if gs.ndim(t) == 0:
                t = gs.expand_dims(t, axis=0)

            if not is_batch:
                return self._simplify_result_t(result.sol(t), space)

            return gs.array(
                [self._simplify_result_t(result.sol(t), space) for result in results]
            )

        return path

    def _simplify_log_result(self, result, space):
        return gs.reshape(result.y[..., 0], (2,) + space.shape)[1]

    def _simplify_result_t(self, result, space):
        return gs.moveaxis(
            gs.reshape(result[: result.shape[0] // 2, :], space.shape + (-1,)), -1, 0
        )
