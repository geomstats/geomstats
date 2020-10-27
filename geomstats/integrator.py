"""Integrator functions used when no closed forms are available.

These are designed for second order ODE written as a first order ODE of two
variables (x,v):
                    dx/dt = v
                    dv/dt = force(x, v)
"""
from functools import partial

from geomstats.errors import check_parameter_accepted_values


def _symplectic_euler_step(state, force, dt):
    """Compute one step of the symplectic euler approximation.

    Parameters
    ----------
    state : array-like, shape=[2, dim]
        variables a time t
    force : callable
    dt : float
        time-step

    Returns
    -------
    point_new : array-like, shape=[dim]
        first variable at time t + dt
    vector_new : array-like, shape=[dim]
        second variable at time t + dt
    """
    point, vector = state
    point_new = point + vector * dt
    vector_new = vector + force(point, vector) * dt
    return point_new, vector_new


def rk4_step(state, force, dt, k1=None):
    """Compute one step of the rk4 approximation.

    Parameters
    ----------
    state : array-like, shape=[2, dim]
        variables a time t
    force : callable
    dt : float
        time-step
    k1 : array-like, shape=[dim]
        initial guess for the slope at time t

    Returns
    -------
    point_new : array-like, shape=[dim]
        first variable at time t + dt
    vector_new : array-like, shape=[dim]
        second variable at time t + dt

    See Also
    --------
    https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """
    point, vector = state
    if k1 is None:
        k1 = force(point, vector)
    k2 = force(point + dt / 2 * vector, vector + dt / 2 * k1)
    k3 = force(point + dt / 2 * vector + dt ** 2 / 2 * k1,
               vector + dt / 2 * k2)
    k4 = force(point + dt * vector + dt ** 2 / 2 * k2, vector + dt * k3)
    point_new = point + dt * vector + dt ** 2 / 6 * (k1 + k2 + k3)
    vector_new = vector + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return point_new, vector_new


def group_rk4_step(group, state, force, dt, k1=None):
    """Compute one step of the rk4 approximation on a Lie group.

    This applies to systems of ODEs where the position variable belongs to a
    Lie group and the velocity variable belongs to its Lie algebra (by
    left-translation).

    Parameters
    ----------
    group : LieGroup
        Lie group on which the integration occurs.
    state : array-like, shape=[2, dim]
        variables a time t
    force : callable
    dt : float
        time-step
    k1 : array-like, shape=[dim]
        initial guess for the slope at time t

    Returns
    -------
    point_new : array-like, shape=[dim]
        first variable at time t + dt
    vector_new : array-like, shape=[dim]
        second variable at time t + dt

    See Also
    --------
    https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """
    point, vector = state
    if k1 is None:
        k1 = group.compose(point, vector)
    l1 = force(point, vector)
    k2 = group.compose(point + dt / 2 * k1, vector + dt / 2 * l1)
    l2 = force(point + dt / 2 * k1, vector + dt / 2 * l1)
    k3 = group.compose(point + dt / 2 * k2, vector + dt / 2 * l2)
    l3 = force(point + dt / 2 * k2, vector + dt / 2 * l2)
    k4 = group.compose(point + dt * k3, vector + dt * l3)
    l4 = force(point + dt * k3, vector + dt * l3)
    point_new = point + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    vector_new = vector + dt / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
    return point_new, vector_new


def group_rk2_step(group, state, force, dt, k1=None):
    """Compute one step of the rk2 approximation on a Lie group.

    This applies to systems of ODEs where the position variable belongs to a
    Lie group and the velocity variable belongs to its Lie algebra (by
    left-translation).

    Parameters
    ----------
    group : LieGroup
        Lie group on which the integration occurs.
    state : array-like, shape=[2, dim]
        variables a time t
    force : callable
    dt : float
        time-step
    k1 : array-like, shape=[dim]
        initial guess for the slope at time t

    Returns
    -------
    point_new : array-like, shape=[dim]
        first variable at time t + dt
    vector_new : array-like, shape=[dim]
        second variable at time t + dt

    See Also
    --------
    https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """
    point, vector = state
    if k1 is None:
        k1 = group.compose(point, vector)
    l1 = force(point, vector)
    k2 = group.compose(point + dt / 2 * k1, vector + dt / 2 * l1)
    l2 = force(point + dt / 2 * k1, vector + dt / 2 * l1)
    point_new = point + dt * k2
    vector_new = vector + dt * l2
    return point_new, vector_new


STEP_FUNCTIONS = {'euler': _symplectic_euler_step,
                  'rk4': rk4_step,
                  'group_rk4': group_rk4_step,
                  'group_rk2': group_rk2_step}


def integrate(
        function, initial_state, end_time=1.0, n_steps=10, step='euler',
        group=None):
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
    group : LieGroup
        If the ODE occurs on a group, it must be passed. Ignored otherwise.
        Optional, default : None.

    Returns
    -------
    final_state : tuple
        sequences of solutions every end_time / n_steps. The shape of each
        element of the sequence is the same as the vectors passed in
        initial_state.
    """
    check_parameter_accepted_values(step, 'step', STEP_FUNCTIONS.keys())

    dt = end_time / n_steps
    positions = [initial_state[0]]
    velocities = [initial_state[1]]
    current_state = (positions[0], velocities[0])

    step_function = STEP_FUNCTIONS[step]
    if 'group' in step and group is not None:
        step_function = partial(step_function, group=group)

    for _ in range(n_steps):
        current_state = step_function(
            state=current_state, force=function, dt=dt)
        positions.append(current_state[0])
        velocities.append(current_state[1])
    return positions, velocities
