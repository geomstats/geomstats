from abc import ABC, abstractmethod

import geomstats.backend as gs


class ExpSolver(ABC):
    @abstractmethod
    def solve(self, metric, tangent_vec, base_point):
        pass

    @abstractmethod
    def geodesic_ivp(self, metric, tangent_vec, base_point, t):
        pass


class ExpIVPSolver(ExpSolver):
    def __init__(self, integrator=None):
        if integrator is None:
            integrator = GSIntegrator()

        self.integrator = integrator

    def _solve(self, metric, tangent_vec, base_point, t_eval=None):
        base_point = gs.broadcast_to(base_point, tangent_vec.shape)

        initial_state = gs.stack([base_point, tangent_vec])

        force = self._get_force(metric)
        if t_eval is None:
            return self.integrator.integrate(force, initial_state)

        return self.integrator.integrate_t(force, initial_state, t_eval)

    def solve(self, metric, tangent_vec, base_point):
        result = self._solve(metric, tangent_vec, base_point)
        return self._simplify_result(result, metric)

    def geodesic_ivp(self, metric, tangent_vec, base_point):

        base_point = gs.broadcast_to(base_point, tangent_vec.shape)
        t_axis = int(len(tangent_vec.shape) > len(metric.shape))

        def path(t):
            squeeze = False
            if not gs.is_array(t):
                t = gs.array([t])
                squeeze = True

            result = self._solve(metric, tangent_vec, base_point, t_eval=t)
            result = self._simplify_result_t(result, metric)
            if squeeze:
                return gs.squeeze(result, axis=t_axis)

            return result

        return path

    def _get_force(self, metric):
        if self.integrator.state_is_raveled:
            force_ = lambda state, t: self._force_raveled_state(state, t, metric=metric)
        else:
            force_ = lambda state, t: self._force_unraveled_state(
                state, t, metric=metric
            )

        if self.integrator.tfirst:
            return lambda t, state: force_(state, t)

        return force_

    def _force_raveled_state(self, raveled_initial_state, _, metric):
        # input: (n,)

        # assumes unvectorize
        state = gs.reshape(raveled_initial_state, (metric.dim, metric.dim))

        # TODO: remove dependency on time in `geodesic_equation`?
        eq = metric.geodesic_equation(state, _)

        return gs.flatten(eq)

    def _force_unraveled_state(self, initial_state, _, metric):
        return metric.geodesic_equation(initial_state, _)

    def _simplify_result(self, result, metric):
        y = result.y[-1]

        if self.integrator.state_is_raveled:
            return y[..., : metric.dim]

        return y[0]

    def _simplify_result_t(self, result, metric):
        # assumes several t
        y = result.y

        if self.integrator.state_is_raveled:
            y = y[..., : metric.dim]
            if gs.ndim(y) > 2:
                return gs.moveaxis(y, 0, 1)
            return y

        y = y[:, 0, :, ...]
        if gs.ndim(y) > 2:
            return gs.moveaxis(y, 1, 0)
        return y


class LogSolver(ABC):
    @abstractmethod
    def solve(self, metric, point, base_point):
        pass


class LogShootingSolver(LogSolver):
    def __init__(self, optimizer=None, initialization=None):
        if optimizer is None:
            optimizer = SCPMinimize()

        if initialization is None:
            initialization = self._default_initialization

        self.optimizer = optimizer
        self.initialization = initialization

    def _default_initialization(self, metric, point, base_point):
        return gs.flatten(gs.random.rand(*base_point.shape))

    def objective(self, velocity, metric, point, base_point):

        velocity = gs.reshape(velocity, base_point.shape)
        delta = metric.exp(velocity, base_point) - point
        return gs.sum(delta**2)

    def solve(self, metric, point, base_point):
        # TODO: are we sure optimizing together is a good idea?

        point, base_point = gs.broadcast_arrays(point, base_point)

        objective = lambda velocity: self.objective(velocity, metric, point, base_point)
        init_tangent_vec = self.initialization(metric, point, base_point)

        res = self.optimizer.optimize(objective, init_tangent_vec, jac="autodiff")

        tangent_vec = gs.reshape(res.x, base_point.shape)

        return tangent_vec


class LogBVPSolver(LogSolver):
    def __init__(self, n_nodes, integrator=None, initialization=None):
        # TODO: add more control on the discretization
        if integrator is None:
            integrator = SCPSolveBVP()

        if initialization is None:
            initialization = self._default_initialization

        self.n_nodes = n_nodes
        self.integrator = integrator
        self.initialization = initialization

    def _default_initialization(self, metric, point, base_point):
        # TODO: receive discretization instead?
        dim = metric.dim
        point_0, point_1 = base_point, point

        # TODO: need to update torch linspace
        # TODO: need to avoid assignment

        lin_init = gs.zeros([2 * dim, self.n_nodes])
        lin_init[:dim, :] = gs.transpose(gs.linspace(point_0, point_1, self.n_nodes))
        lin_init[dim:, :-1] = self.n_nodes * (lin_init[:dim, 1:] - lin_init[:dim, :-1])
        lin_init[dim:, -1] = lin_init[dim:, -2]
        return lin_init

    def boundary_condition(self, state_0, state_1, metric, point_0, point_1):
        pos_0 = state_0[: metric.dim]
        pos_1 = state_1[: metric.dim]
        return gs.hstack((pos_0 - point_0, pos_1 - point_1))

    def bvp(self, _, raveled_state, metric):
        # inputs: n (2*dim) , n_nodes

        # assumes unvectorized

        state = gs.moveaxis(
            gs.reshape(raveled_state, (metric.dim, metric.dim, -1)), -2, -1
        )

        eq = metric.geodesic_equation(state, _)

        eq = gs.reshape(gs.moveaxis(eq, -2, -1), (2 * metric.dim, -1))

        return eq

    def solve(self, metric, point, base_point):
        # TODO: vectorize
        # TODO: assume known jacobian

        bvp = lambda t, state: self.bvp(t, state, metric)
        bc = lambda state_0, state_1: self.boundary_condition(
            state_0, state_1, metric, base_point, point
        )

        x = gs.linspace(0.0, 1.0, self.n_nodes)
        y = self.initialization(metric, point, base_point)

        result = self.integrator.integrate(bvp, bc, x, y)

        return self._simplify_result(result, metric)

    def _simplify_result(self, result, metric):
        _, tangent_vec = gs.reshape(gs.transpose(result.y)[0], (metric.dim, metric.dim))

        return tangent_vec
