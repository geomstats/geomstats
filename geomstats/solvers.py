import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy

import geomstats.backend as gs
import geomstats.integrator as gs_integrator
from geomstats.errors import check_parameter_accepted_values


def _result_to_backend_type(ode_result):
    if gs.__name__.endswith("numpy"):
        return ode_result

    for key, value in ode_result.items():
        if type(value) is np.ndarray:
            ode_result[key] = gs.array(value)

    return ode_result


class OdeResult(scipy.optimize.OptimizeResult):
    # following scipy
    pass


class ODEIVPSolver(metaclass=ABCMeta):
    def __init__(self, save_result=False, state_is_raveled=False, tfirst=False):
        self.state_is_raveled = state_is_raveled
        self.tfirst = tfirst
        self.save_result = save_result

        self.result_ = None

    @abstractmethod
    def integrate(self, force, initial_state, end_time):
        pass

    @abstractmethod
    def integrate_t(self, force, initial_state, t_eval):
        pass


class GSIntegrator(ODEIVPSolver):
    # TODO: control time
    def __init__(self, n_steps=10, step_type="euler", save_result=False):
        super().__init__(save_result=save_result, state_is_raveled=False, tfirst=False)
        self.step_type = step_type
        self.n_steps = n_steps

    @property
    def step_type(self):
        return self._step_type

    @step_type.setter
    def step_type(self, value):
        if callable(value):
            step_function = value
            value = None
        else:
            check_parameter_accepted_values(
                value, "step_type", gs_integrator.STEP_FUNCTIONS
            )
            step_function = getattr(gs_integrator, gs_integrator.STEP_FUNCTIONS[value])

        self._step_function = step_function
        self._step_type = value

    def step(self, force, state, time, dt):
        return self._step_function(force, state, time, dt)

    def _get_n_fevals(self, n_steps):
        n_evals_step = gs_integrator.FEVALS_PER_STEP[self.step_type]
        return n_evals_step * n_steps

    def _integrate(self, force, initial_state, end_time=1.0):
        dt = end_time / self.n_steps
        states = [initial_state]
        current_state = initial_state

        for i in range(self.n_steps):
            current_state = self.step(
                force=force, state=current_state, time=i * dt, dt=dt
            )
            states.append(current_state)

        return states

    def integrate(self, force, initial_state, end_time=1.0):
        states = self._integrate(force, initial_state, end_time=end_time)

        ts = gs.linspace(0.0, end_time, self.n_steps + 1)
        nfev = self._get_n_fevals(self.n_steps)

        result = OdeResult(t=ts, y=gs.array(states), nfev=nfev, njev=0, sucess=True)

        if self.save_result:
            self.result_ = result

        return result

    def integrate_t(self, force, initial_state, t_eval):
        # TODO: this is a very naive implementation
        # based on previous generic implementation in geomstats
        # resolution gets worst for larger t

        states = []
        initial_states = [
            gs.stack([initial_state[0], t * initial_state[1]]) for t in t_eval
        ]
        for initial_state_ in initial_states:
            states_t = self._integrate(force, initial_state_, end_time=1.0)
            states.append(states_t[-1])

        nfev = self._get_n_fevals(self.n_steps)
        n_t = len(t_eval)
        result = OdeResult(
            t=t_eval, y=gs.stack(states), nfev=n_t * nfev, njev=0, sucess=True
        )

        if self.save_result:
            self.result_ = result

        return result


class SCPSolveIVP(ODEIVPSolver):
    # TODO: remember `vectorized` argument
    # TODO: remember `dense_output` argument

    def __init__(self, method="RK45", save_result=False, **options):
        super().__init__(save_result=save_result, state_is_raveled=True, tfirst=True)
        self.method = method
        self.options = options

    def _integrate(self, force, initial_state, end_time=1.0, t_eval=None):
        # TODO: parallelize
        n_points = gs.shape(initial_state)[0] if gs.ndim(initial_state) > 2 else 1

        if n_points > 1:
            results = []
            for position, velocity in zip(*initial_state):
                initial_state_ = gs.stack([position, velocity])
                results.append(
                    self._integrate_single_point(
                        force, initial_state_, end_time, t_eval
                    )
                )

            result = self._merge_results(results)

        else:
            result = self._integrate_single_point(
                force, initial_state, end_time, t_eval=t_eval
            )

        if self.save_result:
            self.result_ = result

        return result

    def integrate(self, force, initial_state, end_time=1.0):
        return self._integrate(force, initial_state, end_time=end_time)

    def integrate_t(self, force, initial_state, t_eval):
        return self._integrate(force, initial_state, end_time=t_eval[-1], t_eval=t_eval)

    def _integrate_single_point(self, force, initial_state, end_time=1.0, t_eval=None):
        raveled_initial_state = gs.flatten(initial_state)

        def force_(t, state):
            state = gs.array(state)
            return force(t, state)

        result = scipy.integrate.solve_ivp(
            force_,
            (0.0, end_time),
            raveled_initial_state,
            method=self.method,
            t_eval=t_eval,
            **self.options
        )
        result = _result_to_backend_type(result)
        result.y = gs.moveaxis(result.y, 0, -1)

        return result

    def _merge_results(self, results):
        # TODO: can "t" and "y" have different shapes?
        keys = ["t", "y", "nfev", "njev", "success"]
        merged_results = {key: [] for key in keys}

        for result in results:
            for key, value in merged_results.items():
                merged_results[key].append(result[key])

        # TODO: should keys other than "t" and "y" be array?
        merged_results = {key: gs.array(value) for key, value in merged_results.items()}
        merged_results["t"] = gs.moveaxis(merged_results["t"], 0, 1)
        merged_results["y"] = gs.moveaxis(merged_results["y"], 0, 1)

        return OdeResult(**merged_results)


class ExpSolver(metaclass=ABCMeta):
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


class SCPMinimize:
    def __init__(self, method="L-BFGS-B", save_result=False, **options):
        self.method = method
        self.options = options
        self.save_result = save_result

        self.options.setdefault("maxiter", 25)

        self.result_ = None

    def optimize(self, func, x0, jac=None):

        if jac == "autodiff":
            jac = True

            def func_(x):
                return gs.autodiff.value_and_grad(func, to_numpy=True)(gs.array(x))

        else:

            def func_(x):
                return func(gs.array(x))

        result = scipy.optimize.minimize(
            func_,
            x0,
            method=self.method,
            jac=jac,
            options=self.options,
        )

        result = _result_to_backend_type(result)

        if result.status == 1:
            logging.warning(
                "Maximum number of iterations reached. Result may be innacurate."
            )

        if self.save_result:
            self.result_ = result

        return result


class SCPSolveBVP:
    def __init__(self, tol=1e-3, max_nodes=1000, save_result=False):
        self.tol = tol
        self.max_nodes = max_nodes
        self.save_result = save_result

        self.result_ = None

    def integrate(self, fun, bc, x, y):
        def bvp(t, state):
            return fun(t, gs.array(state))

        def bc_(state_0, state_1):
            return fun(gs.array(state_0), gs.array(state_1))

        result = scipy.integrate.solve_bvp(
            bvp, bc, x, y, tol=self.tol, max_nodes=self.max_nodes
        )

        result = _result_to_backend_type(result)
        if self.save_result:
            self.result_ = result

        return result


class LogSolver(metaclass=ABCMeta):
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
