def _symplectic_euler_step(state, force, dt):
    """Compute one step of the symplectic euler approximation.

    Parameters
    ----------
    state
    force
    dt

    Returns
    -------
    point_new
    vector_new
    """
    point, vector = state
    point_new = point + vector * dt
    vector_new = vector + force(point, vector) * dt
    return point_new, vector_new


def rk4_step(state, force, dt, k1=None):
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
    function: callable
        the vector field to integrate
    initial_state: tuple
        initial position and speed
    end_time: scalar
    n_steps: int

    Returns
    -------
    a tuple of sequences of solutions every end_time / n_steps
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

