r"""Integrator functions used when no closed forms are available.

These are designed for first order ODE written of a variable x and a time
variable t:
.. math::

                    \frac{dx}{dt} = force(x, t)

where :math: `x` is called the state variable. It may represent many
variables by stacking arrays, e.g. position and velocity in a geodesic
equation.
"""
from scipy.integrate import odeint

import geomstats.backend as gs
from geomstats.errors import check_parameter_accepted_values

METHODS = {"euler": "euler_step", "rk4": "rk4_step", "rk2": "rk2_step",
           "scipy": "scipy"}


def euler_step(force, state, time, dt):
    """Compute one step of the euler approximation.

    Parameters
    ----------
    force : callable
        Vector field that is being integrated.
    state : array-like, shape=[2, dim]
        State at time t, corresponds to position and velocity variables at
        time t.
    time ; float
        Time variable.
    dt : float
        Time-step in the integration.

    Returns
    -------
    point_new : array-like, shape=[,,,, {dim, [n, n]}]
        First variable at time t + dt.
    vector_new : array-like, shape=[,,,, {dim, [n, n]}]
        Second variable at time t + dt.
    """
    derivatives = force(state, time)
    new_state = state + derivatives * dt
    return new_state


def rk2_step(force, state, time, dt):
    """Compute one step of the rk2 approximation.

    Parameters
    ----------
    force : callable
        Vector field that is being integrated.
    state : array-like, shape=[2, dim]
        State at time t, corresponds to position and velocity variables at
        time t.
    time ; float
        Time variable.
    dt : float
        Time-step in the integration.

    Returns
    -------
    point_new : array-like, shape=[,,,, {dim, [n, n]}]
        First variable at time t + dt.
    vector_new : array-like, shape=[,,,, {dim, [n, n]}]
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
    point_new : array-like, shape=[,,,, {dim, [n, n]}]
        First variable at time t + dt.
    vector_new : array-like, shape=[,,,, {dim, [n, n]}]
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


def integrate(function, initial_state, end_time=1.0, n_steps=10, method="euler"):
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
    method : str, {'euler', 'rk4', 'group_rk2', 'group_rk4', 'scipy'}
        Numerical scheme to use for elementary integration steps.
        Optional, default : 'euler'.

    Returns
    -------
    states : array-like, shape=[n_steps + 1, 2, dim]
        contains solutions every end_time / n_steps.
        states[:, 0, :] gives the positions.
        states[:, 1, :] gives the velocities.
    """
    check_parameter_accepted_values(method, "method", METHODS)

    if method == "scipy":
        dim = initial_state[0].shape[0]

        def ivp(state, _):
            """Reformat the initial value problem."""
            position, velocity = state[:dim], state[dim:]
            state = gs.stack([position, velocity])
            vel, acc = function(state, _)
            eq = (vel, acc)
            return gs.hstack(eq)

        initial_state = gs.hstack(initial_state)
        t_int = gs.linspace(0., end_time, n_steps + 1)
        states = odeint(ivp, initial_state, t_int)
        states = states.reshape(n_steps + 1, 2, dim)
    else:
        dt = end_time / n_steps
        states = [initial_state]
        current_state = initial_state

        step_function = globals()[METHODS[method]]

        for i in range(n_steps):
            current_state = step_function(
                state=current_state, force=function, time=i * dt, dt=dt
            )
            states.append(current_state)
        states = gs.stack(states)
    return states
