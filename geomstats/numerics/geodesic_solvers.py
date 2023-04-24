from abc import ABC, abstractmethod

import geomstats.backend as gs
from geomstats.numerics.bvp_solvers import ScipySolveBVP
from geomstats.numerics.ivp_solvers import GSIntegrator
from geomstats.numerics.optimizers import ScipyMinimize


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

            if gs.ndim(t) == 0:
                t = gs.expand_dims(t, axis=0)

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
        state = gs.reshape(raveled_initial_state, (2, space.dim))

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

            if gs.ndim(t) == 0:
                t = gs.expand_dims(t, axis=0)

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
    def __init__(self, optimizer=None, initialization=None):
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
        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        objective = lambda velocity: self._objective(velocity, space, point, base_point)
        init_tangent_vec = self.initialization(space, point, base_point)

        res = self.optimizer.minimize(objective, init_tangent_vec)

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
        return point - base_point

    def _objective(self, velocity, space, point, base_point):
        delta = space.metric.exp(velocity, base_point) - point
        return gs.sum(delta**2)

    def _log_single(self, space, point, base_point):
        objective = lambda velocity: self._objective(velocity, space, point, base_point)
        init_tangent_vec = self.initialization(space, point, base_point)

        res = self.optimizer.minimize(objective, init_tangent_vec)

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
        _, tangent_vec = gs.reshape(gs.transpose(result.y)[0], (2, space.dim))
        return tangent_vec

    def _simplify_result_t(self, result, space):
        return gs.transpose(result[: space.dim, :])


class _Polynomial:
    # TODO: add tests against numpy

    def __init__(self, coeffs):
        self.coeffs = coeffs

    @property
    def degree(self):
        return self.coeffs.shape[-1] - 1

    def __call__(self, x):
        x_t = gs.stack(
            [gs.power(x, power) for power in range(self.degree + 1)], axis=-1
        )
        return gs.einsum("...j,nj->n...", self.coeffs, x_t)

    def first_derivative(self, x):
        dx_t = gs.stack(
            [power * gs.power(x, power - 1) for power in range(1, self.degree + 1)],
            axis=-1,
        )
        return gs.einsum("...j,nj->n...", self.coeffs[..., 1:], dx_t)


class LogPolynomialApproxSolver(_LogBatchMixins, LogSolver):
    def __init__(self, n_nodes=10, degree=5, optimizer=None, initialization=None):
        if optimizer is None:
            optimizer = ScipyMinimize()

        if initialization is None:
            initialization = self._default_initialization

        self.n_nodes = n_nodes
        self.degree = degree
        self.initialization = initialization
        self.optimizer = optimizer

        self._poly = _Polynomial(coeffs=None)

    def _default_initialization(self, space, point, base_point):
        return gs.zeros(space.dim * (self.degree - 1))

    def _create_grid(self):
        return gs.linspace(0.0, 1.0, num=self.n_nodes)

    def _get_coeffs(self, coeffs_mid, space, point, base_point):
        coeffs_0 = gs.transpose(gs.expand_dims(base_point, axis=0))
        coeffs_last = gs.transpose(
            gs.expand_dims(point - base_point - gs.sum(coeffs_mid, axis=-1), axis=0)
        )
        return gs.hstack([coeffs_0, coeffs_mid, coeffs_last])

    def _update_poly_coeffs(self, coeffs_mid, space, point, base_point):
        coeffs_mid = gs.reshape(coeffs_mid, (space.dim, self.degree - 1))
        coeffs = self._get_coeffs(coeffs_mid, space, point, base_point)
        self._poly.coeffs = coeffs
        return self._poly

    def _cost_fun(self, poly, t, space):
        """Compute the energy of the polynomial curve defined by param.

        Parameters
        ----------
        poly : _Polynomial
            Polynomial object with coefficients.
        t : array-like

        Returns
        -------
        energy : float
        """
        point = poly(t)
        tangent_vec = poly.first_derivative(t)

        velocity_sqnorm = space.metric.squared_norm(
            vector=tangent_vec, base_point=point
        )
        return gs.sum(velocity_sqnorm) / t.shape[-1]

    def _objective(self, coeffs_mid, grid, space, point, base_point):
        """Compute function to minimize."""
        poly = self._update_poly_coeffs(coeffs_mid, space, point, base_point)
        return self._cost_fun(poly, grid, space)

    def _solve(self, space, point, base_point):
        x0 = self.initialization(space, point, base_point)
        grid = self._create_grid()

        objective = lambda x: self._objective(
            x, grid=grid, space=space, point=point, base_point=base_point
        )

        return self.optimizer.minimize(objective, x0)

    def _log_single(self, space, point, base_point):
        res = self._solve(space, point, base_point)
        return self._simplify_log_result(res, space, point, base_point)

    def _simplify_log_result(self, result, space, point, base_point):
        poly = self._update_poly_coeffs(result.x, space, point, base_point)
        log = poly.first_derivative(gs.array([1.0]))

        if point.ndim == 1:
            return log[0]

        return log

    def geodesic_bvp(self, space, point, base_point):
        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        is_batch = point.ndim > space.point_ndim
        if not is_batch:
            result = self._solve(space, point, base_point)
            coeffs = self._update_poly_coeffs(result.x, space, point, base_point).coeffs

        else:
            results = [
                self._solve(space, point_, base_point_)
                for point_, base_point_ in zip(point, base_point)
            ]
            coeffs = gs.stack(
                [
                    self._update_poly_coeffs(
                        result.x, space, point_, base_point_
                    ).coeffs
                    for result, point_, base_point_ in zip(results, point, base_point)
                ]
            )

        poly = _Polynomial(coeffs)

        def path(t):
            if not gs.is_array(t):
                t = gs.array([t])

            if gs.ndim(t) == 0:
                t = gs.expand_dims(t, axis=0)

            return poly(t)

        return path
