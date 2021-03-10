r"""Integrator functions used when no closed forms are available.

These are designed for second order ODE written as a first order ODE of two
variables (x,v):
.. math::

                    \frac{dx}{dt} = force_1(x, v)
                    \frac{dv}{dt} = force_2(x, v)

where :math: `x` is called the position variable, :math: `v` the velocity
variable, and :math: `(x, v)` the state.
"""

from geomstats.errors import check_parameter_accepted_values


STEP_FUNCTIONS = {'euler': 'euler_step',
                  'rk4': 'rk4_step',
                  'rk2': 'rk2_step'}


def euler_step(state, force, dt):
    """Compute one step of the euler approximation.

    Parameters
    ----------
    state : array-like, shape=[2, dim]
        State at time t, corresponds to position and velocity variables at
        time t.
    force : callable
        Vector field that is being integrated.
    dt : float
        Time-step in the integration.

    Returns
    -------
    point_new : array-like, shape=[,,,, {dim, [n, n]}]
        First variable at time t + dt.
    vector_new : array-like, shape=[,,,, {dim, [n, n]}]
        Second variable at time t + dt.
    """
    point, vector = state
    velocity, acceleration = force(point, vector)
    point_new = point + velocity * dt
    vector_new = vector + acceleration * dt
    return point_new, vector_new


def rk2_step(state, force, dt):
    """Compute one step of the rk2 approximation.

    Parameters
    ----------
    state : array-like, shape=[2, dim]
        State at time t, corresponds to position and velocity variables at
        time t.
    force : callable
        Vector field that is being integrated.
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
    point, vector = state
    k1, l1 = force(point, vector)
    k2, l2 = force(point + dt / 2 * k1, vector + dt / 2 * l1)
    point_new = point + dt * k2
    vector_new = vector + dt * l2
    return point_new, vector_new


def rk4_step(state, force, dt):
    """Compute one step of the rk4 approximation.

    Parameters
    ----------
    state : array-like, shape=[2, dim]
        State at time t, corresponds to position and velocity variables at
        time t.
    force : callable
        Vector field that is being integrated.
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
    point, vector = state
    k1, l1 = force(point, vector)
    k2, l2 = force(point + dt / 2 * k1, vector + dt / 2 * l1)
    k3, l3 = force(
        point + dt / 2 * k2, vector + dt / 2 * l2)
    k4, l4 = force(point + dt * k3, vector + dt * l3)
    point_new = point + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    vector_new = vector + dt / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
    return point_new, vector_new


def integrate(
        function, initial_state, end_time=1.0, n_steps=10, step='euler'):
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
    check_parameter_accepted_values(step, 'step', STEP_FUNCTIONS)

    dt = end_time / n_steps
    positions = [initial_state[0]]
    velocities = [initial_state[1]]
    current_state = (positions[0], velocities[0])

    step_function = globals()[STEP_FUNCTIONS[step]]

    for _ in range(n_steps):
        current_state = step_function(
            state=current_state, force=function, dt=dt)
        positions.append(current_state[0])
        velocities.append(current_state[1])
    return positions, velocities
