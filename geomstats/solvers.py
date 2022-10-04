from abc import ABCMeta, abstractmethod

import numpy as np
import scipy

import geomstats.backend as gs
import geomstats.integrator as gs_integrator
from geomstats.errors import check_parameter_accepted_values


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


class GSIntegrator(ODEIVPSolver):
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

    def integrate(self, force, initial_state, end_time=1.0):
        dt = end_time / self.n_steps
        states = [initial_state]
        current_state = initial_state

        for i in range(self.n_steps):
            current_state = self.step(
                force=force, state=current_state, time=i * dt, dt=dt
            )
            states.append(current_state)

        ts = gs.linspace(0.0, end_time, self.n_steps + 1)
        nfev = self._get_n_fevals(self.n_steps)

        result = OdeResult(t=ts, y=gs.array(states), nfev=nfev, njev=0, sucess=True)

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

    def integrate(self, force, initial_state, end_time=1.0):
        # TODO: parallelize
        n_points = gs.shape(initial_state)[0] if gs.ndim(initial_state) > 2 else 1

        if n_points > 1:
            results = []
            for position, velocity in zip(*initial_state):
                initial_state_ = gs.stack([position, velocity])
                results.append(self._integrate_single(force, initial_state_, end_time))

            result = self._merge_results(results)

        else:
            result = self._integrate_single(force, initial_state, end_time)

        if self.save_result:
            self.result_ = result

        return result

    def _integrate_single(self, force, initial_state, end_time=1.0):
        # TODO: need to handle single vs multiple point
        # TODO: possible to solve at different time steps (great for geodesic)
        raveled_initial_state = gs.flatten(initial_state)

        def force_(t, state):
            state = gs.array(state)
            return force(t, state)

        result = scipy.integrate.solve_ivp(
            force_,
            (0.0, end_time),
            raveled_initial_state,
            method=self.method,
            **self.options
        )
        result = self._ode_result_to_backend_type(result)
        result.y = gs.moveaxis(result.y, 0, -1)

        return result

    def _ode_result_to_backend_type(self, ode_result):
        if gs.__name__.endswith("numpy"):
            return ode_result

        for key, value in ode_result.items():
            if type(value) is np.ndarray:
                ode_result[key] = gs.array(value)

        return ode_result

    def _merge_results(self, results):
        keys = ["t", "y", "nfev", "njev", "success"]
        merged_results = {key: [] for key in keys}

        for result in results:
            for key, value in merged_results.items():
                merged_results[key].append(result[key])

        merged_results = {key: gs.array(value) for key, value in merged_results.items()}
        merged_results["t"] = gs.moveaxis(merged_results["t"], 0, 1)
        merged_results["y"] = gs.moveaxis(merged_results["y"], 0, 1)

        return OdeResult(**merged_results)


class ExpSolver(metaclass=ABCMeta):
    @abstractmethod
    def solve(self, metric, tangent_vec, base_point):
        pass


class ExpODESolver(ExpSolver):
    # TODO: need to handle vectorization and check for matrix-valued manifolds
    def __init__(self, integrator=None):
        if integrator is None:
            integrator = GSIntegrator()

        self.integrator = integrator

    def solve(self, metric, tangent_vec, base_point):
        base_point = gs.broadcast_to(base_point, tangent_vec.shape)

        initial_state = gs.stack([base_point, tangent_vec])

        force = self._get_force(metric)
        result = self.integrator.integrate(force, initial_state)

        return self._simplify_result(result, metric)

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
        # assumes unvectorized
        position = raveled_initial_state[: metric.dim]
        velocity = raveled_initial_state[metric.dim:]

        state = gs.stack([position, velocity])
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


class SCPMinimize:
    def __init__(self, method="L-BFGS-B", **options):
        # TODO: add save result
        self.method = method
        self.options = options

        self.options.setdefault("maxiter", 25)

    def optimize(self, func, x0, jac=True):

        result = scipy.optimize.minimize(
            func,
            x0,
            method=self.method,
            jac=jac,
            options=self.options,
        )

        return result


class LogSolver(metaclass=ABCMeta):
    @abstractmethod
    def solve(self, metric, point, base_point):
        pass


class LogShootingSolver(LogSolver):
    def __init__(self, optimizer=None):
        if optimizer is None:
            optimizer = SCPMinimize()

        self.optimizer = optimizer

    def objective(self, velocity, metric, point, base_point):
        delta = metric.exp(velocity, base_point) - point
        return gs.sum(delta**2)

    def jacobian(self, objective):
        return gs.autodiff.jacobian(objective)

    def solve(self, metric, point, base_point):
        # TODO: vectorize here?
        # TODO: ensure it works with all backends

        objective = lambda velocity: self.objective(velocity, metric, point, base_point)

        jacobian = self.jacobian(objective)
        tangent_vec = gs.random.rand(*base_point.shape)

        result = self.optimizer.optimize(objective, tangent_vec, jac=jacobian)

        return result
