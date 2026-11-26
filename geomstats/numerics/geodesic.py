"""Geodesic solvers implementation."""

import math
from abc import ABC, abstractmethod

import geomstats.backend as gs
from geomstats.numerics.bvp import ScipySolveBVP
from geomstats.numerics.ivp import GSIVPIntegrator
from geomstats.numerics.optimization import ScipyMinimize
from geomstats.numerics.path import (
    UniformlySampledDiscretePath,
    UniformlySampledPathEnergy,
)
from geomstats.vectorization import get_batch_shape


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
    def exp(self, tangent_vec, base_point):
        """Exponential map.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        end_point : array-like, shape=[..., dim]
            Point on the manifold.
        """

    def geodesic_ivp(self, tangent_vec, base_point):
        """Geodesic curve for initial value problem.

        Parameters
        ----------
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

    Integrate geodesic equation.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    integrator : ODEIVPIntegrator
        Instance of ODEIVP integrator.
    """

    def __init__(self, space, integrator=None):
        self._space = space
        super().__init__()

        if integrator is None:
            integrator = GSIVPIntegrator()

        self._integrator = None
        self.integrator = integrator

    @property
    def integrator(self):
        """An instance of ODEIVPIntegrator."""
        return self._integrator

    @integrator.setter
    def integrator(self, integrator):
        """Set integrator."""
        self.solves_ivp = integrator.tchosen
        self._integrator = integrator

    def _solve(self, tangent_vec, base_point, t_eval=None):
        base_point, tangent_vec = gs.broadcast_arrays(base_point, tangent_vec)

        state_axis = -(self._space.point_ndim + 1)
        initial_state = gs.stack([base_point, tangent_vec], axis=state_axis)

        force = lambda state, _: self._space.connection.geodesic_equation(state)
        if t_eval is None:
            return self.integrator.integrate(force, initial_state)

        return self.integrator.integrate_t(force, initial_state, t_eval)

    def exp(self, tangent_vec, base_point):
        """Exponential map.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.

        Returns
        -------
        end_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        """
        result = self._solve(tangent_vec, base_point)
        return self._simplify_exp_result(result)

    def geodesic_ivp(self, tangent_vec, base_point):
        """Geodesic curve for initial value problem.

        Parameters
        ----------
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
                t = gs.array(t)
            t = gs.expand_dims_to(t, ndim=1)

            result = self._solve(tangent_vec, base_point, t_eval=t)
            return self._simplify_result_t(result)

        return path

    def _simplify_exp_result(self, result):
        y = result.get_last_y()
        point_ndim_slc = tuple([slice(None)] * self._space.point_ndim)
        return y[(..., 0) + point_ndim_slc]

    def _simplify_result_t(self, result):
        # assumes several t
        y = result.y
        point_ndim_slc = tuple([slice(None)] * self._space.point_ndim)
        return y[(..., slice(None), 0) + point_ndim_slc]


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
        point : array-like, shape=[..., dim]
            Point on the manifold.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        """

    def geodesic_bvp(self, point, base_point):
        """Geodesic curve for boundary value problem.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
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
    def _log_single(self, point, base_point):
        """Logarithm map.

        Parameters
        ----------
        point : array-like, shape=[dim]
            Point on the manifold.
        base_point : array-like, shape=[dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[dim]
            Tangent vector at the base point.
        """

    def log(self, point, base_point):
        """Logarithm map.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
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

        is_batch = point.ndim > self._space.point_ndim
        if not is_batch:
            return self._log_single(point, base_point)

        return gs.stack(
            [
                self._log_single(point_, base_point_)
                for point_, base_point_ in zip(point, base_point)
            ]
        )


class LogShootingSolver:
    """Geodesic boundary value problem solver using shooting.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    optimizer : Minimizer
        Optimizer to solve minimization problem.
    initialization : callable
        Function to provide initial solution. `f(point, base_point)`.
        Defaults to linear initialization.
    flatten : bool
        If True, the optimization problem is solved together for all the batch points.
    """

    def __new__(cls, space, optimizer=None, initialization=None, flatten=True):
        """Instantiate a log shooting solver."""
        if flatten:
            return _LogShootingSolverFlatten(
                space=space,
                optimizer=optimizer,
                initialization=initialization,
            )

        return _LogShootingSolverUnflatten(
            space=space,
            optimizer=optimizer,
            initialization=initialization,
        )


class _LogShootingSolver(LogSolver, ABC):
    """Geodesic boundary value problem solver using shooting.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    optimizer : Minimizer
        Optimizer to solve minimization problem.
    initialization : callable
        Function to provide initial solution. `f(point, base_point)`.
        Defaults to linear initialization.
    """

    def __init__(self, space, optimizer=None, initialization=None):
        super().__init__(solves_bvp=False)
        self._space = space

        if optimizer is None:
            optimizer = ScipyMinimize(autodiff_jac=True)

        if initialization is None:
            initialization = self._default_initialization

        self.optimizer = optimizer
        self.initialization = initialization

    def _default_initialization(self, point, base_point):
        """Linear initialization.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        """
        return gs.flatten(point - base_point)

    def _objective(self, flat_tangent_vec, point, base_point, batch_shape):
        """Objective function.

        Minimizes Euclidean distance to shooted point.

        Parameters
        ----------
        flat_tangent_vec : array-like, shape=[prod(batch_shape)*prod(space.shape)]
            Flattened tangent vector.
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        batch_shape : tuple
            Batch shape.
        """
        tangent_vec = gs.reshape(flat_tangent_vec, batch_shape + self._space.shape)
        delta = self._space.connection.exp(tangent_vec, base_point) - point
        return gs.sum(delta**2)


class _LogShootingSolverFlatten(_LogShootingSolver):
    """Geodesic boundary value problem solver using shooting.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    optimizer : Minimizer
        Optimizer to solve minimization problem.
    initialization : callable
        Function to provide initial solution. `f(point, base_point)`.
        Defaults to linear initialization.

    Notes
    -----
    Differs from ``_LogShootingSolverUnflatten`` as it always solves one
    optimization problem, even in batch mode.
    """

    def log(self, point, base_point):
        """Logarithm map.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at the base point.
        """
        batch_shape = get_batch_shape(self._space.point_ndim, point, base_point)

        objective = lambda velocity: self._objective(
            velocity, point, base_point, batch_shape
        )
        init_tangent_vec = self.initialization(point, base_point)

        res = self.optimizer.minimize(objective, init_tangent_vec)

        return gs.reshape(res.x, batch_shape + self._space.shape)


class _LogShootingSolverUnflatten(_LogBatchMixins, _LogShootingSolver):
    """Geodesic boundary value problem solver using shooting.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    optimizer : Minimizer
        Optimizer to solve minimization problem.
    initialization : callable
        Function to provide initial solution. `f(point, base_point)`.
        Defaults to linear initialization.

    Notes
    -----
    Differs from ``_LogShootingSolverFlatten`` as it solves one
    optimization problem for each combination of point and base point.
    """

    def _log_single(self, point, base_point):
        """Logarithm map.

        Parameters
        ----------
        point : array-like, shape=[*space.shape]
            Point on the manifold.
        base_point : array-like, shape=[*space.shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[*space.shape]
            Tangent vector at the base point.
        """
        objective = lambda velocity: self._objective(
            velocity, point, base_point, batch_shape=()
        )
        init_tangent_vec = self.initialization(point, base_point)

        res = self.optimizer.minimize(objective, init_tangent_vec)

        if self._space.point_ndim > 1:
            return gs.reshape(res.x, self._space.shape)

        return res.x


class LogODESolver(_LogBatchMixins, LogSolver):
    """Geodesic boundary value problem using an ODE solver.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    n_nodes : Number of mesh nodes.
    integrator : ScipySolveBVP
        Instance of ScipySolveBVP.
    initialization : callable
        Function to provide initial solution. `f( point, base_point)`.
        Defaults to linear initialization.
    """

    def __init__(
        self, space, n_nodes=10, integrator=None, initialization=None, use_jac=True
    ):
        self._space = space
        super().__init__(solves_bvp=True)

        if integrator is None:
            integrator = ScipySolveBVP()

        if initialization is None:
            initialization = self._default_initialization

        self.n_nodes = n_nodes
        self.integrator = integrator
        self.initialization = initialization
        if use_jac and space.point_ndim != 1:
            raise ValueError(
                "The Jacobian of the geodesic equation is only implemented for "
                "points represented by a vector. Use `use_jac=False`."
            )
        self.use_jac = use_jac

        self.grid = self._create_grid()

    def _create_grid(self):
        return gs.linspace(0.0, 1.0, num=self.n_nodes)

    def _default_initialization(self, point, base_point):
        if point.ndim == 1:
            point_0, point_1 = base_point, point
        else:
            point_0 = gs.flatten(base_point)
            point_1 = gs.flatten(point)

        pos_init = gs.matrix_transpose(gs.linspace(point_0, point_1, self.n_nodes))

        vel_init = self.n_nodes * (pos_init[:, 1:] - pos_init[:, :-1])
        vel_init = gs.hstack([vel_init, vel_init[:, [-2]]])

        return gs.vstack([pos_init, vel_init])

    def _boundary_condition(self, state_0, state_1, point_0, point_1):
        pos_0 = state_0[: point_0.shape[0]]
        pos_1 = state_1[: point_1.shape[0]]
        return gs.hstack((pos_0 - point_0, pos_1 - point_1))

    def _bvp(self, raveled_state):
        """Boundary value problem.

        Parameters
        ----------
        raveled_state : array-like, shape=[2*dim, n_nodes]
            Vector of state variables (position and speed).

        Returns
        -------
        sol : array-like, shape=[2*dim, n_nodes]
        """
        state = gs.moveaxis(
            gs.reshape(raveled_state, (2,) + self._space.shape + (-1,)), -1, 0
        )
        new_state = self._space.connection.geodesic_equation(state)
        return gs.moveaxis(gs.reshape(new_state, (-1, raveled_state.shape[0])), -2, -1)

    def _jacobian(self, _, raveled_state):
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
        dim = self._space.dim
        n_nodes = raveled_state.shape[-1]
        position = gs.moveaxis(raveled_state[:dim], -1, 0)
        velocity = raveled_state[dim:]

        dgamma = self._space.connection.jacobian_christoffels(position)

        df_dposition = -gs.einsum(
            "j...,...ijkl,k...->il...", velocity, dgamma, velocity
        )

        gamma = self._space.connection.christoffels(position)
        df_dvelocity = -2 * gs.einsum("...ijk,k...->ij...", gamma, velocity)

        jac_nw = gs.zeros((dim, dim, raveled_state.shape[1]))
        jac_ne = gs.broadcast_to(
            gs.expand_dims(gs.eye(dim), axis=-1), (dim, dim, n_nodes)
        )

        jac_sw = df_dposition
        jac_se = df_dvelocity
        return gs.concatenate(
            (
                gs.concatenate((jac_nw, jac_ne), axis=1),
                gs.concatenate((jac_sw, jac_se), axis=1),
            ),
            axis=0,
        )

    def _solve(self, point, base_point):
        bvp = lambda t, state: self._bvp(state)
        bc = lambda state_0, state_1: self._boundary_condition(
            state_0, state_1, gs.flatten(base_point), gs.flatten(point)
        )

        jacobian = None
        if self.use_jac:
            jacobian = lambda t, state: self._jacobian(t, state)

        y = self.initialization(point, base_point)

        return self.integrator.integrate(bvp, bc, self.grid, y, fun_jac=jacobian)

    def _log_single(self, point, base_point):
        res = self._solve(point, base_point)
        return self._simplify_log_result(res)

    def geodesic_bvp(self, point, base_point):
        """Geodesic curve for boundary value problem.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point on the manifold.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        path : callable
            Time parametrized geodesic curve. `f(t)`. 0 <= t <= 1.
        """
        point, base_point = gs.broadcast_arrays(point, base_point)

        is_batch = point.ndim > self._space.point_ndim
        if not is_batch:
            result = self._solve(point, base_point)
        else:
            results = [
                self._solve(point_, base_point_)
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
                t = gs.array(t)
            t = gs.expand_dims_to(t, ndim=1)

            if not is_batch:
                return self._simplify_result_t(result.sol(t))

            return gs.stack(
                [self._simplify_result_t(result.sol(t)) for result in results]
            )

        return path

    def _simplify_log_result(self, result):
        return gs.reshape(result.y[..., 0], (2,) + self._space.shape)[1]

    def _simplify_result_t(self, result):
        return gs.moveaxis(
            gs.reshape(result[: result.shape[0] // 2, :], self._space.shape + (-1,)),
            -1,
            0,
        )


class _DiscreteGeodesicBVPBatchMixins:
    @abstractmethod
    def _discrete_geodesic_bvp_single(self, point, base_point):
        """Solve boundary value problem (BVP).

        Given an initial point and an end point, solve the geodesic equation
        via minimizing the Riemannian path energy.

        Parameters
        ----------
        point : array-like, shape=[*point_shape]
            Point on the manifold.
        base_point : array-like, shape=[*point_shape]
            Point on the manifold.

        Returns
        -------
        discr_geod_path : array-like, shape=[n_nodes, *point_shape]
            Discrete geodesic.
        """

    def discrete_geodesic_bvp(self, point, base_point):
        """Solve boundary value problem (BVP).

        Given an initial point and an end point, solve the geodesic equation
        via minimizing the Riemannian path energy.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        discr_geod_path : array-like, shape=[..., n_nodes, *point_shape]
            Discrete geodesic.
        """
        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        is_batch = point.ndim > self._space.point_ndim
        if not is_batch:
            return self._discrete_geodesic_bvp_single(point, base_point)

        return gs.stack(
            [
                self._discrete_geodesic_bvp_single(point_, base_point_)
                for point_, base_point_ in zip(point, base_point)
            ]
        )


class PathBasedLogSolver(LogSolver, ABC):
    """A geodesic BVP solver based on finding a discrete geodesic path.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    """

    def __init__(self, space):
        self._space = space
        super().__init__(solves_bvp=True)

    def path_n_nodes(self, path):
        """Get number of nodes of a path.

        Parameters
        ----------
        path : array-like, shape=[..., n_nodes, *point_shape]
            Discrete path.

        Returns
        -------
        n_nodes : int
            Number of path nodes.
        """
        return path.shape[-self._space.point_ndim - 1]

    @abstractmethod
    def discrete_geodesic_bvp(self, point, base_point):
        """Solve boundary value problem (BVP).

        Given an initial point and an end point, solve the geodesic equation
        via minimizing the Riemannian path energy.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        discr_geod_path : array-like, shape=[..., n_nodes, *point_shape]
            Discrete geodesic.
        """

    def log(self, point, base_point):
        """Logarithm map.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at the base point.
        """
        discr_geod_path = self.discrete_geodesic_bvp(point, base_point)
        point_ndim_slc = (slice(None),) * self._space.point_ndim
        return (self.path_n_nodes(discr_geod_path) - 1) * (
            discr_geod_path[(..., 1) + point_ndim_slc]
            - discr_geod_path[(..., 0) + point_ndim_slc]
        )

    def geodesic_bvp(self, point, base_point):
        """Geodesic curve for boundary value problem.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        path : callable
            Time parametrized geodesic curve. `f(t)`.
        """
        discr_geod_path = self.discrete_geodesic_bvp(point, base_point)
        return UniformlySampledDiscretePath(
            discr_geod_path, point_ndim=self._space.point_ndim
        )


class PathStraightening(_DiscreteGeodesicBVPBatchMixins, PathBasedLogSolver):
    """Geodesic boundary value problem with path-straightening.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    path_energy : callable
        Method to compute Riemannian path energy.
    n_nodes : int
        Number of path discretization points.
    optimizer : Minimizer
        An optimizer to solve path energy minimization problem.
    initialization : callable
        A method to get initial guess for optimization.
    symmetric : bool
        If to use a symmetrized version of the energy.

    References
    ----------
    .. [HSKCB2022] "Elastic shape analysis of surfaces with second-order
        Sobolev metrics: a comprehensive numerical framework".
        arXiv:2204.04238 [cs.CV], 25 Sep 2022
    """

    def __init__(
        self,
        space,
        path_energy=None,
        n_nodes=100,
        optimizer=None,
        initialization=None,
        symmetric=False,
    ):
        super().__init__(space)
        if optimizer is None:
            optimizer = ScipyMinimize(
                method="L-BFGS-B",
                autodiff_jac=True,
                options={"disp": False},
            )

        if path_energy is None:
            path_energy = UniformlySampledPathEnergy(space)

        if initialization is None:
            initialization = self._default_initialization

        self.n_nodes = n_nodes
        self.optimizer = optimizer
        self.path_energy = path_energy
        self.initialization = initialization
        self.symmetric = symmetric

    def _default_initialization(self, point, base_point):
        """Linear initialization.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        path : array-like, shape=[..., n_nodes, *point_shape]
            Linear path between base point and point.
        """
        times = gs.linspace(0.0, 1.0, self.n_nodes)
        linear_deformation = point - base_point
        return base_point + gs.einsum("t,...->t...", times, linear_deformation)

    def _make_objective(self, point, base_point):
        """Create objective function.

        Parameters
        ----------
        point : array-like, shape=[*point_shape]
            Point on the manifold.
        base_point : array-like, shape=[*point_shape]
            Point on the manifold.

        Returns
        -------
        objective : callable
            Objective function.
        """

        def objective(midpoints):
            """Compute path energy of paths going through a midpoint.

            Parameters
            ----------
            midpoints : array-like, shape=[(self.n_nodes-2) * math.prod(*point_shape)]
                Midpoints of the path.

            Returns
            -------
            _ : float
                Energy of the path.
            """
            midpoints = gs.reshape(midpoints, (self.n_nodes - 2,) + self._space.shape)
            path = gs.concatenate(
                [
                    base_point,
                    midpoints,
                    point,
                ],
            )
            energy = self.path_energy(path)
            if self.symmetric:
                reversed_path = gs.flip(path, axis=0)
                energy = (energy + self.path_energy(reversed_path)) / 2

            return energy

        return objective

    def _discrete_geodesic_bvp_single(self, point, base_point):
        """Solve boundary value problem (BVP).

        Given an initial point and an end point, solve the geodesic equation
        via minimizing the Riemannian path energy.

        Parameters
        ----------
        point : array-like, shape=[*point_shape]
            Point on the manifold.
        base_point : array-like, shape=[*point_shape]
            Point on the manifold.

        Returns
        -------
        discr_geod_path : array-like, shape=[n_nodes, *point_shape]
            Discrete geodesic.
        """
        init_path = self.initialization(point, base_point)
        init_midpoints = init_path[1:-1]

        base_point = gs.expand_dims(base_point, axis=0)
        point = gs.expand_dims(point, axis=0)

        init_midpoints = gs.reshape(init_midpoints, (-1,))

        objective = self._make_objective(point, base_point)
        sol = self.optimizer.minimize(objective, init_midpoints)

        solution_midpoints = gs.reshape(
            gs.array(sol.x), (self.n_nodes - 2,) + self._space.shape
        )

        return gs.concatenate(
            [
                base_point,
                solution_midpoints,
                point,
            ],
            axis=0,
        )


class MultiresPathStraightening(_DiscreteGeodesicBVPBatchMixins, PathBasedLogSolver):
    """Geodesic boundary value problem with multiresolution path straightening.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    path_energy : callable
        Method to compute Riemannian path energy.
    n_nodes : list
        Number of path discretization points at each resolution.
    n_levels : int
        Number of resolutions to use. Sets number of nodes following a sequence.
        Ignored if ``n_nodes`` is not None.
    optimizer : Minimizer
        An optimizer to solve path energy minimization problem.
    initialization : callable
        A method to get initial guess for optimization.
    symmetric : bool
        If to use a symmetrized version of the energy.
    early_stop : bool
        If to stop at a resolution if previous resolution was solved
        with one iteration.
    """

    def __init__(
        self,
        space,
        path_energy=None,
        n_nodes=None,
        n_levels=3,
        optimizer=None,
        initialization=None,
        symmetric=False,
        early_stop=False,
    ):
        super().__init__(space)
        if optimizer is None:
            optimizer = ScipyMinimize(
                method="L-BFGS-B",
                autodiff_jac=True,
                options={"disp": False},
            )

        if n_nodes is None:
            n_nodes = self._create_levels_sequence(3, n_levels)

        if path_energy is None:
            path_energy = UniformlySampledPathEnergy(space)

        if initialization is None:
            initialization = self._default_initialization

        self.n_nodes = n_nodes
        self.optimizer = optimizer
        self.path_energy = path_energy
        self.initialization = initialization
        self.symmetric = symmetric
        self.early_stop = early_stop

    @staticmethod
    def _create_levels_sequence(init_val, n_levels):
        seq = [init_val]
        for _ in range(n_levels - 1):
            seq.append(seq[-1] * 2 - 1)

        return seq

    def _default_initialization(self, point, base_point):
        """Linear initialization.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        path : array-like, shape=[..., n_nodes, *point_shape]
            Linear path between base point and point.
        """
        times = gs.linspace(0.0, 1.0, self.n_nodes[0])
        linear_deformation = point - base_point
        return base_point + gs.einsum("t,...->t...", times, linear_deformation)

    def _initialization_from_previous(self, path, n_nodes):
        """Initialize next resolution using previous resolution solution.

        Parameters
        ----------
        path : array-like, shape=[..., n_nodes_previous, *point_shape]
            Discrete path.

        Returns
        -------
        interp_path : array-like, shape=[..., n_nodes, *point_shape]
            Discrete path.
        """
        discr_geod_path = UniformlySampledDiscretePath(
            path, point_ndim=self._space.point_ndim
        )
        times = gs.linspace(0.0, 1.0, n_nodes)
        return discr_geod_path(times)

    def _make_objective(self, point, base_point, n_nodes):
        """Create objective function.

        Parameters
        ----------
        point : array-like, shape=[*point_shape]
            Point on the manifold.
        base_point : array-like, shape=[*point_shape]
            Point on the manifold.

        Returns
        -------
        objective : callable
            Objective function.
        """

        def objective(midpoints):
            """Compute path energy of paths going through a midpoint.

            Parameters
            ----------
            midpoints : array-like, shape=[(n_nodes-2) * math.prod(*point_shape)]
                Midpoints of the path.

            Returns
            -------
            _ : float
                Energy of the path.
            """
            midpoints = gs.reshape(midpoints, (n_nodes - 2,) + self._space.shape)
            path = gs.concatenate(
                [
                    base_point,
                    midpoints,
                    point,
                ],
            )
            energy = self.path_energy(path)
            if self.symmetric:
                reversed_path = gs.flip(path, axis=0)
                energy = (energy + self.path_energy(reversed_path)) / 2

            return energy

        return objective

    def _discrete_geodesic_bvp_single(self, point, base_point):
        """Solve boundary value problem (BVP).

        Given an initial point and an end point, solve the geodesic equation
        via minimizing the Riemannian path energy.

        Parameters
        ----------
        point : array-like, shape=[*point_shape]
            Point on the manifold.
        base_point : array-like, shape=[*point_shape]
            Point on the manifold.

        Returns
        -------
        discr_geod_path : array-like, shape=[n_nodes, *point_shape]
            Discrete geodesic.
        """
        init_path = self.initialization(point, base_point)
        init_midpoints = init_path[1:-1]

        base_point = gs.expand_dims(base_point, axis=0)
        point = gs.expand_dims(point, axis=0)

        discrete_path = None
        for n_nodes in self.n_nodes:
            objective = self._make_objective(point, base_point, n_nodes)

            if discrete_path is None:
                init_midpoints = self.initialization(point, base_point)[1:-1]

            else:
                init_midpoints = self._initialization_from_previous(
                    discrete_path, n_nodes
                )[1:-1]

            init_midpoints = gs.reshape(init_midpoints, (-1,))
            res = self.optimizer.minimize(objective, init_midpoints)

            path_midpoints = gs.reshape(
                gs.array(res.x), (n_nodes - 2,) + self._space.shape
            )
            discrete_path = gs.concatenate(
                [
                    base_point,
                    path_midpoints,
                    point,
                ],
                axis=0,
            )

            if self.early_stop and res.nit == 1:
                break

        return discrete_path


class PolynomialPathLogSolver(
    _LogBatchMixins, _DiscreteGeodesicBVPBatchMixins, PathBasedLogSolver
):
    """Geodesic boundary value problem restricted to polynomial paths.

    The coordinates of the path are polynomial functions of time of degree
    ``degree``, with fixed values at times 0 and 1. The free coefficients
    minimize the energy of the path, approximated by its average squared
    speed at ``n_times`` uniformly sampled times.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    degree : int
        Degree of the polynomial functions of time.
    n_times : int
        Number of times at which the path is sampled.
    optimizer : Minimizer
        Optimizer to solve the energy minimization problem.
    """

    def __init__(self, space, degree=5, n_times=200, optimizer=None):
        super().__init__(space)

        if optimizer is None:
            optimizer = ScipyMinimize(method="BFGS", autodiff_jac=gs.has_autodiff())

        self.degree = degree
        self.n_times = n_times
        self.optimizer = optimizer

    def _coefficients(self, free_coefficients, point, base_point):
        """Assemble the coefficients of the path.

        Parameters
        ----------
        free_coefficients : array-like, shape=[degree - 1, *point_shape]
            Coefficients of the monomials of degree 1 to degree - 1.
        point : array-like, shape=[*point_shape]
            End point.
        base_point : array-like, shape=[*point_shape]
            Initial point.

        Returns
        -------
        coefficients : array-like, shape=[degree + 1, *point_shape]
            Coefficients of the monomials of degree 0 to degree.
        """
        last_coefficient = point - base_point - gs.sum(free_coefficients, axis=0)
        return gs.concatenate(
            [
                gs.expand_dims(base_point, axis=0),
                free_coefficients,
                gs.expand_dims(last_coefficient, axis=0),
            ],
            axis=0,
        )

    def _path(self, coefficients, times):
        """Evaluate the path.

        Parameters
        ----------
        coefficients : array-like, shape=[degree + 1, *point_shape]
            Coefficients of the monomials of degree 0 to degree.
        times : array-like, shape=[n_times,]
            Times.

        Returns
        -------
        path : array-like, shape=[n_times, *point_shape]
            Points of the path.
        """
        powers = gs.stack([times**power for power in range(self.degree + 1)])
        return gs.einsum("kn,k...->n...", powers, coefficients)

    def _velocity(self, coefficients, times):
        """Evaluate the velocity of the path.

        Parameters
        ----------
        coefficients : array-like, shape=[degree + 1, *point_shape]
            Coefficients of the monomials of degree 0 to degree.
        times : array-like, shape=[n_times,]
            Times.

        Returns
        -------
        velocity : array-like, shape=[n_times, *point_shape]
            Velocities of the path.
        """
        powers = gs.stack(
            [power * times ** (power - 1) for power in range(1, self.degree + 1)]
        )
        return gs.einsum("kn,k...->n...", powers, coefficients[1:])

    def _solve_single(self, point, base_point):
        """Find the coefficients of the polynomial geodesic.

        Parameters
        ----------
        point : array-like, shape=[*point_shape]
            End point.
        base_point : array-like, shape=[*point_shape]
            Initial point.

        Returns
        -------
        coefficients : array-like, shape=[degree + 1, *point_shape]
            Coefficients of the monomials of degree 0 to degree.
        """
        free_shape = (self.degree - 1,) + self._space.shape
        times = gs.linspace(0.0, 1.0, self.n_times)

        def objective(flat_free_coefficients):
            free_coefficients = gs.reshape(flat_free_coefficients, free_shape)
            coefficients = self._coefficients(free_coefficients, point, base_point)
            path = self._path(coefficients, times)
            velocity = self._velocity(coefficients, times)
            return gs.mean(self._space.metric.squared_norm(velocity, path))

        init_free_coefficients = gs.zeros(math.prod(free_shape))
        sol = self.optimizer.minimize(objective, init_free_coefficients)

        free_coefficients = gs.reshape(gs.array(sol.x), free_shape)
        return self._coefficients(free_coefficients, point, base_point)

    def _discrete_geodesic_bvp_single(self, point, base_point):
        """Solve boundary value problem (BVP).

        Parameters
        ----------
        point : array-like, shape=[*point_shape]
            End point.
        base_point : array-like, shape=[*point_shape]
            Initial point.

        Returns
        -------
        discr_geod_path : array-like, shape=[n_times, *point_shape]
            Discrete geodesic.
        """
        coefficients = self._solve_single(point, base_point)
        return self._path(coefficients, gs.linspace(0.0, 1.0, self.n_times))

    def _log_single(self, point, base_point):
        """Logarithm map.

        The initial velocity of a polynomial path is its coefficient of degree 1.

        Parameters
        ----------
        point : array-like, shape=[*point_shape]
            End point.
        base_point : array-like, shape=[*point_shape]
            Initial point.

        Returns
        -------
        tangent_vec : array-like, shape=[*point_shape]
            Tangent vector at the base point.
        """
        return self._solve_single(point, base_point)[1]
