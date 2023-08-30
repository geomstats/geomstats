r"""Illustrate how a Kalman-like filter can be defined on Lie groups.

A generic Kalman filter class is defined for systems on Lie groups for which
the exponential is surjective. Its use is illustrated on two localization
problems, a linear and a non-linear one. In both cases, the propagation model
is known, and sparse position measurements are obtained. It thus relies on a
`model` of the system, providing the system's equations and jacobians.

The former is a 1D-localization problem where the state is a 2D vector (x, v)
made of the system's position and speed. The process writes
:math:`(x_{i+1}, v_{i+1}) = (x_i + dt * v_i, v_i + dt * a_i)`,
where dt is the time-step between i and i+1, and a_i a noisy acceleration
measured by a given sensor.

The latter is a 2D pose (position + orientation) estimation problem, where the
state (R, x) is made a planar rotation and 2D position, i.e. a member of SE(2).
The non-linear propagation writes
:math:`(R_{i+1}, x_{i+1}) = (R_i \Omega_i, x_i + dt * R_i u_i)`,
where :math:`\Omega_i, u_i` is the measured odometry of the system.
The implementation follows that of the Invariant Extended Kalman Filter (IEKF)
which was designed for such cases [BB2017].

References
----------
.. [BB2017] Barrau, Bonnabel, "The Invariant Extended Kalman Filter as a Stable
Observer", IEEE Transactions on Automatic Control, 2017
https://arxiv.org/abs/1410.1465
"""

import matplotlib.pyplot as plt
import numpy as np

import geomstats.backend as gs
from geomstats import algebra_utils
from geomstats.learning.kalman_filter import (
    KalmanFilter,
    Localization,
    LocalizationLinear,
)


def create_data(kalman, true_init, true_inputs, obs_freq):
    """Create data for a specific example.

    Parameters
    ----------
    kalman : KalmanFilter
        Filter which will be used to estimate the state.
    true_init : array-like, shape=[dim]
        True initial state.
    true_inputs : list(array-like, shape=[dim_input])
        Noise-free inputs giving the evolution of the true state.
    obs_freq : int
        Number of time steps between observations.

    Returns
    -------
    true_traj : array-like, shape=[len(true_inputs), dim]
        Trajectory of the true state.
    inputs : list(array-like, shape=[dim_input])
        Simulated noisy inputs received by the sensor.
    observations : array-like, shape=[len(true_inputs)/obs_freq, dim_obs]
        Simulated noisy observations of the system.
    """
    true_traj = [1 * true_init]
    for incr in true_inputs:
        true_traj.append(kalman.model.propagate(true_traj[-1], incr))
    true_obs = [
        kalman.model.observation_model(pose) for pose in true_traj[obs_freq::obs_freq]
    ]

    obs_dtype = true_obs[0].dtype
    observations = gs.stack(
        [
            gs.array(
                np.random.multivariate_normal(obs, kalman.measurement_noise),
                dtype=obs_dtype,
            )
            for obs in true_obs
        ]
    )

    input_dtype = true_inputs[0].dtype
    inputs = [
        gs.concatenate(
            [
                incr[:1],
                gs.array(
                    np.random.multivariate_normal(incr[1:], kalman.process_noise),
                    dtype=input_dtype,
                ),
            ],
            axis=0,
        )
        for incr in true_inputs
    ]
    inputs = [gs.cast(incr, input_dtype) for incr in inputs]

    return gs.array(true_traj), inputs, observations


def estimation(kalman, initial_state, inputs, observations, obs_freq):
    """Carry out the state estimation for a specific system.

    Parameters
    ----------
    kalman : KalmanFilter
        Filter used to estimate the state.
    initial_state : array-like, shape=[dim]
        Guess of the true initial state.
    inputs : list(array-like, shape=[dim_input])
        Inputs received by the propagation sensor.
    observations : array-like, shape=[len(inputs) + 1/obs_freq, dim_obs]
        Measurements of the system.
    obs_freq : int
        Number of time steps between observations.

    Returns
    -------
    traj : array-like, shape=[len(inputs) + 1, dim]
        Estimated trajectory.
    three_sigmas : array-like, shape=[len(inputs) + 1, dim]
        3-sigma envelope of the estimated state covariance.
    """
    kalman.state = 1 * initial_state

    traj = [1 * kalman.state]
    uncertainty = [1 * gs.diagonal(kalman.covariance)]
    for i, _ in enumerate(inputs):
        kalman.propagate(inputs[i])
        if i > 0 and i % obs_freq == obs_freq - 1:
            kalman.update(observations[(i // obs_freq)])
        traj.append(1 * kalman.state)
        uncertainty.append(1 * gs.diagonal(kalman.covariance))
    traj = gs.array(traj)
    uncertainty = gs.array(uncertainty)
    three_sigmas = 3 * gs.sqrt(uncertainty)

    return traj, three_sigmas


def main():
    """Carry out two examples of state estimation on groups.

    Both examples are localization problems, where only a part of the system
    is observed. The first one is a linear system, while the second one is
    non-linear.
    """
    np.random.seed(12345)
    model = LocalizationLinear()
    kalman = KalmanFilter(model)
    n_traj = 1000
    obs_freq = 50
    dt = 0.1
    init_cov = gs.array([10.0, 1.0])
    init_cov = algebra_utils.from_vector_to_diagonal_matrix(init_cov)
    prop_cov = 0.001 * gs.eye(model.dim_noise)
    obs_cov = 10 * gs.eye(model.dim_obs)
    initial_covs = (init_cov, prop_cov, obs_cov)
    kalman.initialize_covariances(*initial_covs)

    true_state = gs.array([0.0, 0.0])
    true_acc = gs.random.uniform(-1, 1, (n_traj, 1))
    dt_vectorized = dt * gs.ones((n_traj, 1))
    true_inputs = gs.hstack((dt_vectorized, true_acc))

    true_traj, inputs, observations = create_data(
        kalman, true_state, true_inputs, obs_freq
    )

    initial_state = np.random.multivariate_normal(true_state, init_cov)
    estimate, uncertainty = estimation(
        kalman, initial_state, inputs, observations, obs_freq
    )

    plt.figure()
    plt.plot(true_traj[:, 0], label="Ground Truth")
    plt.plot(estimate[:, 0], label="Kalman")
    plt.plot(estimate[:, 0] + uncertainty[:, 0], color="k", linestyle=":")
    plt.plot(
        estimate[:, 0] - uncertainty[:, 0],
        color="k",
        linestyle=":",
        label="3_sigma envelope",
    )
    plt.plot(
        range(obs_freq, n_traj + 1, obs_freq),
        observations,
        marker="*",
        linestyle="",
        label="Observation",
    )
    plt.legend()
    plt.title("1D Localization - Position")

    plt.figure()
    plt.plot(true_traj[:, 1], label="Ground Truth")
    plt.plot(estimate[:, 1], label="Kalman")
    plt.plot(estimate[:, 1] + uncertainty[:, 1], color="k", linestyle=":")
    plt.plot(
        estimate[:, 1] - uncertainty[:, 1],
        color="k",
        linestyle=":",
        label="3_sigma envelope",
    )
    plt.legend()
    plt.title("1D Localization - Speed")

    model = Localization()
    kalman = KalmanFilter(model)
    init_cov = gs.array([1.0, 10.0, 10.0])
    init_cov = algebra_utils.from_vector_to_diagonal_matrix(init_cov)
    prop_cov = 0.001 * gs.eye(model.dim_noise)
    obs_cov = 0.1 * gs.eye(model.dim_obs)
    initial_covs = (init_cov, prop_cov, obs_cov)
    kalman.initialize_covariances(*initial_covs)

    true_state = gs.zeros(model.group.dim)
    true_inputs = [gs.array([dt, 0.5, 0.0, 0.05]) for _ in range(n_traj)]

    true_traj, inputs, observations = create_data(
        kalman, true_state, true_inputs, obs_freq
    )

    initial_state = gs.array(np.random.multivariate_normal(true_state, init_cov))
    initial_state = gs.cast(initial_state, true_state.dtype)
    estimate, uncertainty = estimation(
        kalman, initial_state, inputs, observations, obs_freq
    )

    plt.figure()
    plt.plot(true_traj[:, 1], true_traj[:, 2], label="Ground Truth")
    plt.plot(estimate[:, 1], estimate[:, 2], label="Kalman")
    plt.scatter(observations[:, 0], observations[:, 1], s=2, c="k", label="Observation")
    plt.legend()
    plt.axis("equal")
    plt.title("2D Localization")

    plt.show()


if __name__ == "__main__":
    main()
