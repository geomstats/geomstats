"""Integrator functions used when no closed forms are available.

These are designed for second order ODE written as a first order ODE of two
variables (x,v):
                    dx/dt = v
                    dv/dt = force(x, v)
"""


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


def integrate(function, initial_state, end_time=1.0, n_steps=10, step='euler'):
    """Compute the flow under the vector field using symplectic euler.

    Integration function to compute flows of vector fields
    on a regular grid between 0 and a finite time from an initial state.

    Parameters
    ----------
    function : callable
        the vector field to integrate
    initial_state : tuple
        initial position and speed
    end_time : scalar
    n_steps : int
    step : str, {euler, rk4}
        the numerical scheme to use for elementary integration steps

    Returns
    -------
    final_state : tuple
        sequences of solutions every end_time / n_steps. The shape of each
        element of the sequence is the same as the vectors passed in
        initial_state.
    """
    dt = end_time / n_steps
    positions = [initial_state[0]]
    velocities = [initial_state[1]]
    current_state = (positions[0], velocities[0])
    step_function = _symplectic_euler_step if step == 'euler' else rk4_step
    for _ in range(n_steps):
        current_state = step_function(current_state, function, dt)
        positions.append(current_state[0])
        velocities.append(current_state[1])
    return positions, velocities
