from abc import ABC, abstractmethod

import geomstats.backend as gs
from geomstats.numerics.bvp_solvers import ScipySolveBVP
from geomstats.numerics.ivp_solvers import GSIntegrator
from geomstats.numerics.optimizers import ScipyMinimize

# TODO: check uses of space.dim


class ExpSolver(ABC):
    @abstractmethod
    def exp(self, space, tangent_vec, base_point):
        pass

    @abstractmethod
    def geodesic_ivp(self, space, tangent_vec, base_point, t):
        pass


class ExpIVPSolver(ExpSolver):
    def __init__(self, integrator=None):
        if integrator is None:
            integrator = GSIntegrator()

        self.integrator = integrator

    def _solve(self, space, tangent_vec, base_point, t_eval=None):
        base_point = gs.broadcast_to(base_point, tangent_vec.shape)

        if self.integrator.state_is_raveled:
            initial_state = gs.hstack([base_point, tangent_vec])
        else:
            initial_state = gs.stack([base_point, tangent_vec])

        force = self._get_force(space)
        if t_eval is None:
            return self.integrator.integrate(force, initial_state)

        return self.integrator.integrate_t(force, initial_state, t_eval)

    def exp(self, space, tangent_vec, base_point):
        result = self._solve(space, tangent_vec, base_point)
        return self._simplify_exp_result(result, space)

    def geodesic_ivp(self, space, tangent_vec, base_point):
        base_point = gs.broadcast_to(base_point, tangent_vec.shape)

        def path(t):
            if not gs.is_array(t):
                t = gs.array([t])

            result = self._solve(space, tangent_vec, base_point, t_eval=t)
            return self._simplify_result_t(result, space)

        return path

    def _get_force(self, space):
        if self.integrator.state_is_raveled:
            force_ = lambda state, t: self._force_raveled_state(state, t, space=space)
        else:
            force_ = lambda state, t: self._force_unraveled_state(state, t, space=space)

        if self.integrator.tfirst:
            return lambda t, state: force_(state, t)

        return force_

    def _force_raveled_state(self, raveled_initial_state, _, space):
        # input: (n,)

        # assumes unvectorize
        state = gs.reshape(raveled_initial_state, (space.dim, space.dim))

        eq = space.metric.geodesic_equation(state, _)

        return gs.flatten(eq)

    def _force_unraveled_state(self, initial_state, _, space):
        return space.metric.geodesic_equation(initial_state, _)

    def _simplify_exp_result(self, result, space):
        y = result.get_last_y()

        if self.integrator.state_is_raveled:
            return y[..., : space.dim]

        return y[0]

    def _simplify_result_t(self, result, space):
        # assumes several t
        y = result.y

        if self.integrator.state_is_raveled:
            y = y[..., : space.dim]

            if gs.ndim(y) > 2:
                return gs.moveaxis(y, 0, 1)
            return y

        y = y[:, 0, :, ...]
        if gs.ndim(y) > 2:
            return gs.moveaxis(y, 1, 0)
        return y


class LogSolver(ABC):
    @abstractmethod
    def log(self, space, point, base_point):
        pass

    @abstractmethod
    def geodesic_bvp(self, space, point, base_point):
        pass


class _GeodesicBVPFromExpMixins:
    def _geodesic_bvp_single(self, space, t, tangent_vec, base_point):
        tangent_vec_ = gs.einsum("...,...i->...i", t, tangent_vec)
        return space.metric.exp(tangent_vec_, base_point)

    def geodesic_bvp(self, space, point, base_point):
        tangent_vec = self.log(space, point, base_point)
        is_batch = tangent_vec.ndim > space.point_ndim

        def path(t):
            if not gs.is_array(t):
                t = gs.array([t])

            if not is_batch:
                return self._geodesic_bvp_single(space, t, tangent_vec, base_point)

            return gs.stack(
                [
                    self._geodesic_bvp_single(space, t, tangent_vec_, base_point_)
                    for tangent_vec_, base_point_ in zip(tangent_vec, base_point)
                ]
            )

        return path


class _LogBatchMixins:
    @abstractmethod
    def _log_single(self, space, point, base_point):
        pass

    def log(self, space, point, base_point):
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
    def __new__(cls, optimizer=None, initialization=None, flatten=True):
        if flatten:
            return _LogShootingSolverFlatten(
                optimizer=optimizer,
                initialization=initialization,
            )

        return _LogShootingSolverUnflatten(
            optimizer=optimizer,
            initialization=initialization,
        )


class _LogShootingSolverFlatten(_GeodesicBVPFromExpMixins, LogSolver):
    # TODO: add a (linear) initialization here?

    def __init__(self, optimizer=None, initialization=None):
        if optimizer is None:
            optimizer = ScipyMinimize(jac="autodiff")

        if initialization is None:
            initialization = self._default_initialization

        self.optimizer = optimizer
        self.initialization = initialization

    def _default_initialization(self, space, point, base_point):
        return gs.flatten(gs.random.rand(*base_point.shape))

    def _objective(self, velocity, space, point, base_point):
        velocity = gs.reshape(velocity, base_point.shape)
        delta = space.metric.exp(velocity, base_point) - point
        return gs.sum(delta**2)

    def log(self, space, point, base_point):
        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        objective = lambda velocity: self._objective(velocity, space, point, base_point)
        init_tangent_vec = self.initialization(space, point, base_point)

        res = self.optimizer.optimize(objective, init_tangent_vec)

        tangent_vec = gs.reshape(res.x, base_point.shape)

        return tangent_vec


class _LogShootingSolverUnflatten(
    _LogBatchMixins, _GeodesicBVPFromExpMixins, LogSolver
):
    def __init__(self, optimizer=None, initialization=None):
        if optimizer is None:
            optimizer = ScipyMinimize(jac="autodiff")

        if initialization is None:
            initialization = self._default_initialization

        self.optimizer = optimizer
        self.initialization = initialization

    def _default_initialization(self, space, point, base_point):
        return gs.random.rand(*base_point.shape)

    def _objective(self, velocity, space, point, base_point):
        delta = space.metric.exp(velocity, base_point) - point
        return gs.sum(delta**2)

    def _log_single(self, space, point, base_point):
        objective = lambda velocity: self._objective(velocity, space, point, base_point)
        init_tangent_vec = self.initialization(space, point, base_point)

        res = self.optimizer.optimize(objective, init_tangent_vec)

        return res.x


class LogBVPSolver(_LogBatchMixins, LogSolver):
    def __init__(self, n_nodes=10, integrator=None, initialization=None):
        if integrator is None:
            integrator = ScipySolveBVP()

        if initialization is None:
            initialization = self._default_initialization

        self.n_nodes = n_nodes
        self.integrator = integrator
        self.initialization = initialization

        self.grid = self._create_grid()

    def _create_grid(self):
        return gs.linspace(0.0, 1.0, num=self.n_nodes)

    def _default_initialization(self, space, point, base_point):
        point_0, point_1 = base_point, point

        pos_init = gs.transpose(gs.linspace(point_0, point_1, self.n_nodes))

        vel_init = self.n_nodes * (pos_init[:, 1:] - pos_init[:, :-1])
        vel_init = gs.hstack([vel_init, vel_init[:, [-2]]])

        return gs.vstack([pos_init, vel_init])

    def boundary_condition(self, state_0, state_1, space, point_0, point_1):
        pos_0 = state_0[: space.dim]
        pos_1 = state_1[: space.dim]
        return gs.hstack((pos_0 - point_0, pos_1 - point_1))

    def bvp(self, _, raveled_state, space):
        # inputs: n (2*dim) , n_nodes
        # assumes unvectorized

        state = gs.moveaxis(gs.reshape(raveled_state, (2, space.dim, -1)), -2, -1)

        eq = space.metric.geodesic_equation(state, _)

        return gs.reshape(gs.moveaxis(eq, -2, -1), (2 * space.dim, -1))

    def _solve(self, space, point, base_point):
        bvp = lambda t, state: self.bvp(t, state, space)
        bc = lambda state_0, state_1: self.boundary_condition(
            state_0, state_1, space, base_point, point
        )

        y = self.initialization(space, point, base_point)

        return self.integrator.integrate(bvp, bc, self.grid, y)

    def _log_single(self, space, point, base_point):
        res = self._solve(space, point, base_point)
        return self._simplify_log_result(res, space)

    def geodesic_bvp(self, space, point, base_point):
        # TODO: add to docstrings: 0 <= t <= 1

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
            if not gs.is_array(t):
                t = gs.array(t)

            if not is_batch:
                return self._simplify_result_t(result.sol(t), space)

            return gs.array(
                [self._simplify_result_t(result.sol(t), space) for result in results]
            )

        return path

    def _simplify_log_result(self, result, space):
        _, tangent_vec = gs.reshape(gs.transpose(result.y)[0], (2, space.dim))
        return tangent_vec
    
    def _simplify_result_t(self, result, space):
        return gs.transpose(result[:space.dim, :])


class LogPolynomialSolver(_LogBatchMixins, LogSolver):

    def __init__(self, n_nodes=201, degree=5, optimizer=None, initialization=None):
        if optimizer is None:
            optimizer = ScipyMinimize()

        if initialization is None:
            initialization = self._default_initialization
        
        self.n_nodes = n_nodes
        self.degree  = degree
        self.initialization = initialization
        self.optimizer = optimizer
    
    def _default_initialization(self, space, point, base_point, param, t=None):
        last_coef = point - base_point - gs.sum(param, axis=0)
        coef = gs.vstack((base_point, param, last_coef))

        t = gs.linspace(0.0, 1.0, self.n_nodes) if t is None else t

        t_x = gs.stack([t**i for i in range(self.degree + 1)])
        x = gs.einsum("ij,ik->kj", coef, t_x)

        t_y = gs.stack([i * t ** (i - 1) for i in range(1, self.degree + 1)])
        y = gs.einsum("ij,ik->kj", coef[1:], t_y)

        return x, y
    
    def _cost_fun(self, param, space, point, base_point):
        """Compute the energy of the polynomial curve defined by param."""

        x, y = self.initialization(space, point, base_point, param)
        if x.min() < 0:
            return np.inf, np.inf, x, np.nan

        velocity_sqnorm = space.metric.squared_norm(vector=y, base_point=x)
        energy = gs.sum(velocity_sqnorm) / self.n_nodes
        return energy, x, y
    
    def _cost_jacobian(self, param, space, point, base_point):
        """Compute the jacobian of the cost function at polynomial curve."""

        t = gs.linspace(0.0, 1.0, self.n_nodes)
        x, y = self.initialization(space, point, base_point, param, t=t)
        
        fac1 = gs.stack(
            [
                k * t ** (k - 1) - self.degree * t ** (self.degree - 1)
                for k in range(1, self.degree)
            ]
        )
        fac2 = gs.stack([t**k - t**self.degree for k in range(1, self.degree)])
        fac3 = (y * gs.polygamma(1, x)).T - gs.sum(
            y, 1
        ) * gs.polygamma(1, gs.sum(x, 1))
        fac4 = (y**2 * gs.polygamma(2, x)).T - gs.sum(
            y, 1
        ) ** 2 * gs.polygamma(2, gs.sum(x, 1))

        cost_jac = (
            2 * gs.einsum("ij,kj->ik", fac1, fac3)
            + gs.einsum("ij,kj->ik", fac2, fac4)
        ) / self.n_nodes
        return gs.reshape(gs.transpose(cost_jac), (space.dim * (self.degree - 1),))

    def _f2minimize(self, x, space, point, base_point):
        """Compute function to minimize."""
        param = gs.transpose(gs.reshape(x,(space.dim, self.degree - 1)))
        return self._cost_fun(param, space, point, base_point)[0]

    def _jacobian(self, x, space, point, base_point):
        """Compute jacobian of the function to minimize."""
        param = gs.transpose(gs.reshape(gs.from_numpy(x),(space.dim, self.degree - 1)))
        return self._cost_jacobian(param, space, point, base_point)
    
    def _solve(
        self,
        space,
        point,
        base_point,
        jac_on=True,
    ):
        
        x0 = gs.ones(space.dim * (self.degree - 1))
        jac = self._jacobian if jac_on else None
        sol = self.optimizer.optimize(self._f2minimize, x0, jac=jac, args=(space, point, base_point))
        opt_param = gs.transpose(gs.reshape(sol.x,(space.dim, self.degree - 1)))
        _, x, y = self._cost_fun(opt_param, space, point, base_point)

        return x, y
    
    def _log_single(self, space, point, base_point):
        _, y = self._solve(space, point, base_point)
        return y[0]
    
    def _simplify_result_t(self, x, t):
        """Get point at t using interpolation"""
        n_segments = x.shape[0]-1
        points_at_t = []
        for t_ in t:
            i1 = int(t_*(n_segments))
            t1 = i1/n_segments
            if i1 == n_segments:
                points_at_t.append(x[i1])
                continue
            i2 = i1+1
            t2 = i2/n_segments
            points_at_t.append(x[i1] + (t_-t1)*(x[i2]-x[i1])/(t2-t1))
        return gs.stack(points_at_t)

    def geodesic_bvp(self, space, point, base_point):
        # TODO: add to docstrings: 0 <= t <= 1

        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        is_batch = point.ndim > space.point_ndim
        if not is_batch:
            result = self._solve(space, point, base_point)[0]
        else:
            results = [
                self._solve(space, point_, base_point_)[0]
                for point_, base_point_ in zip(point, base_point)
            ]

        def path(t):
            if not is_batch:
                return self._simplify_result_t(result, t)

            return gs.array(
                [self._simplify_result_t(result, t) for result in results]
            )

        return path
    
