r"""Integrator functions used when no closed forms are available.

Lead author: Nicolas Guigui.

These are designed for first order ODE written of a variable x and a time
variable t:

.. math::

    \frac{dx}{dt} = force(x, t)

where :math:`x` is called the state variable. It may represent many
variables by stacking arrays, e.g. position and velocity in a geodesic
equation.
"""

from geomstats.errors import check_parameter_accepted_values

STEP_FUNCTIONS = {
    "euler": "euler_step",
    "rk4": "rk4_step",
    "rk2": "rk2_step",
}


FEVALS_PER_STEP = {
    "euler": 1,
    "rk4": 4,
    "rk2": 2,
}


def euler_step(force, state, time, dt):
    """Compute one step of the euler approximation.

    Parameters
    ----------
    force : callable
        Vector field that is being integrated.
    state : array-like, shape=[2, dim]
        State at time t, corresponds to position and velocity variables at
        time t.
    time : float
        Time variable.
    dt : float
        Time-step in the integration.

    Returns
    -------
    point_new : array-like, shape=[..., {dim, [n, n]}]
        First variable at time t + dt.
    vector_new : array-like, shape=[..., {dim, [n, n]}]
        Second variable at time t + dt.
    """
    derivatives = force(state, time)
    new_state = state + derivatives * dt
    return new_state


def symplectic_euler_step(force, state, time, dt):
    """Compute one step of the symplectic euler approximation.

    Parameters
    ----------
    state : array-like, shape=[2, dim]
        State at time t, corresponds to position and velocity variables at
        time t.
    force : callable
        Vector field that is being integrated.
    time : float
        Time variable.
    dt : float
        Time-step in the integration.

    Returns
    -------
    point_new : array-like, shape=[..., {dim, [n, n]}]
        First variable at time t + dt.
    vector_new : array-like, shape=[..., {dim, [n, n]}]
        Second variable at time t + dt.
    """
    raise NotImplementedError


def leapfrog_step(force, state, time, dt):
    """Compute one step of the leapfrog approximation.

    Parameters
    ----------
    state : array-like, shape=[2, dim]
        State at time t, corresponds to position and velocity variables at
        time t.
    force : callable
        Vector field that is being integrated.
    time : float
        Time variable.
    dt : float
        Time-step in the integration.

    Returns
    -------
    point_new : array-like, shape=[..., {dim, [n, n]}]
        First variable at time t + dt.
    vector_new : array-like, shape=[..., {dim, [n, n]}]
        Second variable at time t + dt.
    """
    raise NotImplementedError


def rk2_step(force, state, time, dt):
    """Compute one step of the rk2 approximation.

    Parameters
    ----------
    force : callable
        Vector field that is being integrated.
    state : array-like, shape=[2, dim]
        State at time t, corresponds to position and velocity variables at
        time t.
    time : float
        Time variable.
    dt : float
        Time-step in the integration.

    Returns
    -------
    point_new : array-like, shape=[..., {dim, [n, n]}]
        First variable at time t + dt.
    vector_new : array-like, shape=[..., {dim, [n, n]}]
        Second variable at time t + dt.

    See Also
    --------
    https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """
    k1 = force(state, time)
    k2 = force(state + dt / 2 * k1, time + dt / 2)
    new_state = state + dt * k2
    return new_state


def rk4_step(force, state, time, dt):
    """Compute one step of the rk4 approximation.

    Parameters
    ----------
    force : callable
        Vector field that is being integrated.
    state : array-like, shape=[2, dim]
        State at time t, corresponds to position and velocity variables at
        time t.
    time : float
        Time variable.
    dt : float
        Time-step in the integration.

    Returns
    -------
    point_new : array-like, shape=[..., {dim, [n, n]}]
        First variable at time t + dt.
    vector_new : array-like, shape=[..., {dim, [n, n]}]
        Second variable at time t + dt.

    See Also
    --------
    https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """
    k1 = force(state, time)
    k2 = force(state + dt / 2 * k1, time + dt / 2)
    k3 = force(state + dt / 2 * k2, time + dt / 2)
    k4 = force(state + dt * k3, time + dt)
    new_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return new_state


def integrate(function, initial_state, end_time=1.0, n_steps=10, step="euler"):
    """Compute the flow under the vector field using symplectic euler.

    Integration function to compute flows of vector fields
    on a regular grid between 0 and a finite time from an initial state.

    Parameters
    ----------
    function : callable
        Vector field to integrate.
    initial_state : tuple of arrays
        Initial position and speed.
    end_time : float
        Final integration time.
        Optional, default : 1.
    n_steps : int
        Number of integration steps to use.
        Optional, default : 10.
    step : str, {'euler', 'rk4', 'group_rk2', 'group_rk4'}
        Numerical scheme to use for elementary integration steps.
        Optional, default : 'euler'.

    Returns
    -------
    final_state : tuple
        sequences of solutions every end_time / n_steps. The shape of each
        element of the sequence is the same as the vectors passed in
        initial_state.
    """
    check_parameter_accepted_values(step, "step", STEP_FUNCTIONS)

    dt = end_time / n_steps
    states = [initial_state]
    current_state = initial_state

    step_function = globals()[STEP_FUNCTIONS[step]]

    for i in range(n_steps):
        current_state = step_function(
            state=current_state, force=function, time=i * dt, dt=dt
        )
        states.append(current_state)
    return states
