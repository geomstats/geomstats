"""Initial value problem solvers implementation."""

from abc import ABC, abstractmethod

import scipy

import geomstats.backend as gs
import geomstats.integrator as gs_integrator
from geomstats.errors import check_parameter_accepted_values
from geomstats.numerics._common import result_to_backend_type


def _merge_scipy_results(results, same_t=False):
    keys = list(results[0].keys())
    merged_results = {key: [] for key in keys}

    for result in results:
        for key, value in merged_results.items():
            value.append(result[key])

    if same_t:
        merged_results["t"] = gs.moveaxis(gs.stack(merged_results["t"]), 0, 1)
        merged_results["y"] = gs.moveaxis(gs.stack(merged_results["y"]), 0, 1)

    return merged_results


class OdeResult(scipy.optimize.OptimizeResult):
    """Bunch object (follows scipy).

    Its purposes is to homogenize output of different integrators.
    """

    def get_last_y(self):
        """Get value for last y.

        Allows to have y represented as an `gs.array` or `list[gs.array]` (latter
        for cases where they have different shapes).

        Assumes last `t` is the same.
        """
        if isinstance(self.y, (list, tuple)):
            return gs.stack([y_[-1] for y_ in self.y])

        return self.y[-1]


class ODEIVPSolver(ABC):
    """Abstract class for ode ivp solvers.

    Parameters
    ----------
    save_result : bool
        If True, result is stored after calling `integrate` or `integrate_t`.
    state_is_raveled : bool
        If True, state is represented by a 1d array.
        Else, it is represented by `(n_vars, dim)`.
    tfirst : bool
        Declares function signature.
        If True `f(t, y)`, else `f(y, t)`.
    tchosen : bool
        Informs about ability to solve at chosen times.
        If False, then does not implement `integrate_t`.
    """

    def __init__(
        self, save_result=False, state_is_raveled=False, tfirst=False, tchosen=False
    ):
        self.state_is_raveled = state_is_raveled
        self.tfirst = tfirst
        self.save_result = save_result
        self.tchosen = tchosen

        self.result_ = None

    @abstractmethod
    def integrate(self, force, initial_state, end_time):
        """Integrate force.

        Parameters
        ----------
        force : callable
            Function to integrate.
        initial_state : array-like
            Initial state.
        end_time : float or None
            Integration end time.

        Returns
        -------
        result : OdeResult
        """

    def integrate_t(self, force, initial_state, t_eval):
        """Integrate force while choosing evaluating points.

        Parameters
        ----------
        force : callable
            Function to integrate.
        initial_state : array-like
            Initial state.
        t_eval : array-like
            Times at which to store the computed solution.

        Returns
        -------
        result : OdeResult
        """
        raise NotImplementedError("Can't solve for chosen evaluating points.")


class GSIVPIntegrator(ODEIVPSolver):
    """In-house ODE integrator.

    Parameters
    ----------
    n_steps : int
        Number of steps to perform.
    step_type : str
        Type of integration step.
        Possible values are `euler`, `rk2`, `rk4`.
    save_result : bool
        If True, result is stored after calling `integrate` or `integrate_t`.
    """

    def __init__(self, n_steps=10, step_type="euler", save_result=False):
        super().__init__(
            save_result=save_result, state_is_raveled=False, tfirst=False, tchosen=False
        )
        self.step_type = step_type
        self.n_steps = n_steps

    @property
    def step_type(self):
        """Integrator step type."""
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

    def _step(self, force, state, time, dt):
        return self._step_function(force, state, time, dt)

    def _get_n_fevals(self, n_steps):
        n_evals_step = gs_integrator.FEVALS_PER_STEP[self.step_type]
        return n_evals_step * n_steps

    def _integrate(self, force, initial_state, end_time=1.0):
        dt = end_time / self.n_steps
        states = [initial_state]
        current_state = initial_state

        for i in range(self.n_steps):
            current_state = self._step(
                force=force, state=current_state, time=i * dt, dt=dt
            )
            states.append(current_state)

        return states

    def integrate(self, force, initial_state, end_time=1.0):
        """Integrate force.

        Parameters
        ----------
        force : callable
            Function to integrate.
        initial_state : array-like
            Initial state.
        end_time : float or None
            Integration end time.

        Returns
        -------
        result : OdeResult
        """
        states = self._integrate(force, initial_state, end_time=end_time)

        ts = gs.linspace(0.0, end_time, self.n_steps + 1)
        nfev = self._get_n_fevals(self.n_steps)

        result = OdeResult(t=ts, y=gs.array(states), nfev=nfev, njev=0, sucess=True)

        if self.save_result:
            self.result_ = result

        return result


class ScipySolveIVP(ODEIVPSolver):
    """Wrapper for scipy.integrate.solve_ivp.

    Parameters
    ----------
    method : str
        Integration method.
    save_result : bool
        If True, result is stored after calling `integrate` or `integrate_t`.
    """

    def __init__(self, method="RK45", save_result=False, **options):
        super().__init__(
            save_result=save_result, state_is_raveled=True, tfirst=True, tchosen=True
        )
        self.method = method
        self.options = options

    def _integrate(self, force, initial_state, end_time=1.0, t_eval=None):
        if initial_state.ndim > 1:
            results = []
            for initial_state_ in initial_state:
                results.append(
                    self._integrate_single(force, initial_state_, end_time, t_eval)
                )

            result = OdeResult(_merge_scipy_results(results, same_t=t_eval is not None))

        else:
            result = self._integrate_single(
                force, initial_state, end_time, t_eval=t_eval
            )
            result = OdeResult(**result)

        if self.save_result:
            self.result_ = result

        return result

    def integrate(self, force, initial_state, end_time=1.0):
        """Integrate force.

        Parameters
        ----------
        force : callable
            Function to integrate.
        initial_state : array-like
            Initial state.
        end_time : float or None
            Integration end time.

        Returns
        -------
        result : OdeResult
        """
        return self._integrate(force, initial_state, end_time=end_time)

    def integrate_t(self, force, initial_state, t_eval):
        """Integrate force at `t_eval` points.

        Parameters
        ----------
        force : callable
            Function to integrate.
        initial_state : array-like
            Initial state.
        t_eval : array-like
            Times at which to store the computed solution.

        Returns
        -------
        result : OdeResult
        """
        return self._integrate(force, initial_state, end_time=t_eval[-1], t_eval=t_eval)

    def _integrate_single(self, force, initial_state, end_time=1.0, t_eval=None):
        def force_(t, state):
            state = gs.from_numpy(state)
            return force(t, state)

        result = scipy.integrate.solve_ivp(
            force_,
            (0.0, end_time),
            initial_state,
            method=self.method,
            t_eval=t_eval,
            **self.options,
        )
        result = result_to_backend_type(result)
        result.y = gs.moveaxis(result.y, 0, -1)

        return result
