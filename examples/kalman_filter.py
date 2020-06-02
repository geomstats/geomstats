r"""Illustrate how a Kalman-like filter can be defined on Lie groups.

A generic Kalman filter class is defined for systems on Lie groups for which
the exponential is subjective. Its use is illustrated on two localization
problems, a linear and a non-linear one. In both cases, the propagation model
is known, and sparse position measurements are obtained. It thus relies on a
`model` of the system, providing the system's equations and jacobians.

The former is a 1D-localization problem where the state is a 2D vector (x, v)
made of the system's position and speed. The process writes
(x_{i+1}, v_{i+1}) = (x_i + dt * v_i, v_i + dt * a_i),
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

import os

import matplotlib.pyplot as plt
import numpy as np

import geomstats.backend as gs
from geomstats import algebra_utils
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.special_euclidean import SpecialEuclidean


class LocalizationLinear:
    """Class for modeling a linear 1D localization problem.

    The state is made of a scalar position and scalar speed, thus a 2D vector.
    A sensor provides acceleration inputs, while another one provides sparse
    measurements of the position.
    """
    group = Euclidean(2)
    dim = group.dim
    dim_noise = 1
    dim_obs = 1

    @staticmethod
    def propagate(state, sensor_input):
        """Propagate with piece-wise constant acceleration and velocity."""
        dt, acc = sensor_input
        pos, speed = state
        pos = pos + dt * speed
        speed = speed + dt * acc
        return gs.array([pos, speed])

    @staticmethod
    def propagation_jacobian(state, sensor_input):
        """Compute the matrix associated to the affine propagation."""
        dt, _ = sensor_input
        dim = LocalizationLinear.dim
        position_line = gs.hstack((gs.eye(dim // 2), dt * gs.eye(dim // 2)))
        speed_line = gs.hstack((
            gs.zeros((dim // 2, dim // 2)), gs.eye(dim // 2)))
        jac = gs.vstack((position_line, speed_line))
        # jac = gs.assignment(jac, dt, (0, 1))
        return jac

    @staticmethod
    def noise_jacobian(state, sensor_input):
        """Compute the matrix associated to the propagation noise.

        The noise is supposed additive.
        """
        dt, _ = sensor_input
        dim = LocalizationLinear.dim
        position_wrt_noise = gs.zeros((dim // 2, dim // 2))
        speed_wrt_noise = gs.sqrt(dt) * gs.eye(dim // 2)
        jac = gs.vstack((position_wrt_noise, speed_wrt_noise))
        return jac

    @staticmethod
    def observation_jacobian(state, observation):
        """Compute the matrix associated to the observation model."""
        return gs.eye(LocalizationLinear.dim_obs, LocalizationLinear.dim)

    @staticmethod
    def get_measurement_noise_cov(state, observation_cov):
        """Get the observation covariance."""
        return observation_cov

    @staticmethod
    def observation_model(state):
        """Model used to create the measurements."""
        return state[:1]

    @staticmethod
    def innovation(state, observation):
        """Discrepancy between the measurement and its expected value."""
        return observation - LocalizationLinear.observation_model(state)


class Localization:
    """Class for modeling a non-linear 2D localization problem.

    The state is composed of a planar orientation and position, and is thus a
    member of SE(2).
    A sensor provides the linear and angular speed, while another one provides
    sparse position observations.
    """
    group = SpecialEuclidean(2, 'vector')
    dim = group.dim
    dim_rot = group.rotations.dim
    dim_noise = 3
    dim_obs = 2

    @staticmethod
    def split_input(sensor_input):
        """Separate the input into its main parts."""
        return sensor_input[0], sensor_input[1:Localization.group.n + 1], sensor_input[Localization.group.n + 1:]

    @staticmethod
    def rotation_matrix(theta):
        """Construct the rotation matrix associated to the angle theta."""
        if gs.ndim(gs.array(theta)) <= 1:
            theta = gs.array([theta])
        return Localization.group.rotations.matrix_from_rotation_vector(theta)

    @staticmethod
    def regularize_angle(theta):
        """Bring back angle theta in ]-pi, pi]."""
        if gs.ndim(gs.array(theta)) < 1:
            theta = gs.array([theta])
        return Localization.group.rotations.log_from_identity(theta)

    @staticmethod
    def adjoint_map(state):
        """Construct the tangent map associated to Ad_X : g |-> XgX^-1."""
        theta, x, y = state
        tangent_base = gs.array([[0., -1.],
                                 [1., 0.]])
        orientation_part = gs.eye(Localization.dim_rot, Localization.dim)
        pos_column = gs.reshape(state[1:], (Localization.group.n, 1))
        position_wrt_orientation = - Matrices.mul(tangent_base, pos_column)
        position_wrt_position = Localization.rotation_matrix(theta)
        last_lines = gs.hstack((
            position_wrt_orientation, position_wrt_position))
        ad = gs.vstack((orientation_part, last_lines))

        return ad

    @staticmethod
    def propagate(state, sensor_input):
        """Propagate state with constant velocity motion model on SE(2)."""
        dt, linear_speed, angular_speed = Localization.split_input(sensor_input)
        theta, x, y = state
        local_speed = Matrices.mul(
            Localization.rotation_matrix(theta), linear_speed)
        new_pos = state[1:] + dt * local_speed
        theta = theta + dt * angular_speed
        theta = Localization.regularize_angle(theta)
        return gs.concatenate((theta, new_pos), axis=0)

    @staticmethod
    def propagation_jacobian(state, sensor_input):
        """Construct the jacobian associated to the input.

        Since the propagation writes f(x) = x*u, and the error is modeled on
        the Lie algebra, the jacobian is Ad_{u^{-1}}.
        """
        dt, linear_speed, angular_speed = Localization.split_input(sensor_input)
        input_vector_form = dt * gs.concatenate((angular_speed, linear_speed), axis=0)
        input_inv = Localization.group.inverse(input_vector_form)

        return Localization.adjoint_map(input_inv)

    @staticmethod
    def noise_jacobian(state, sensor_input):
        """Construct the jacobian associated to the process noise.

        The noise being considered multiplicative, it is simply the identity
        scaled by the time stamp.
        """
        dt, _, _ = Localization.split_input(sensor_input)
        return gs.sqrt(dt) * gs.eye(Localization.dim_noise)

    @staticmethod
    def observation_jacobian(state, observation):
        """Construct the jacobian associated to the innovation."""
        orientation_part = gs.zeros((Localization.dim_obs, Localization.dim_rot))
        position_part = gs.eye(Localization.dim_obs, Localization.group.n)
        return gs.hstack((orientation_part, position_part))

    @staticmethod
    def get_measurement_noise_cov(state, observation_cov):
        """Construct the measurement covariance for the innovation."""
        theta, _, _ = state
        rot = Localization.rotation_matrix(theta)
        return Matrices.mul(Matrices.transpose(rot), observation_cov, rot)

    @staticmethod
    def observation_model(state):
        """Model used to obtain the measurements."""
        return state[1:]

    @staticmethod
    def innovation(state, observation):
        """Discrepancy between the measurement and its expected value.

        Here the linear error observation - expected is brought back to
        the Lie algebra
        """
        theta, _, _ = state
        rot = Localization.rotation_matrix(theta)
        expected = Localization.observation_model(state)
        return Matrices.mul(Matrices.transpose(rot), observation - expected)


class KalmanFilter:
    """Class for a general Kalman filter working on Lie groups.

    Given an adapted model, it provides the tools to carry out non-linear state
    estimation with an error modeled on the Lie algebra.
    """

    def __init__(self, model):
        self.model = model
        self.state = model.group.get_identity()
        self.covariance = gs.zeros((self.model.dim, self.model.dim))
        self.process_noise = gs.zeros(
            (self.model.dim_noise, self.model.dim_noise))
        self.measurement_noise = gs.zeros(
            (self.model.dim_obs, self.model.dim_obs))

    def initialize_covariances(self, prior_values, process_values, obs_values):
        """Set the values of the covariances."""
        values = [prior_values, process_values, obs_values]
        attributes = ['covariance', 'process_noise', 'measurement_noise']
        for (index, val) in enumerate(values):
            setattr(self, attributes[index], val)

    def propagate(self, sensor_input):
        """Propagate the estimate and its covariance."""
        prop_noise = self.process_noise
        prop_jac = self.model.propagation_jacobian(self.state, sensor_input)
        noise_jac = self.model.noise_jacobian(self.state, sensor_input)

        prop_cov = Matrices.mul(
            prop_jac, self.covariance, Matrices.transpose(prop_jac))
        noise_cov = Matrices.mul(
            noise_jac, prop_noise, Matrices.transpose(noise_jac))
        self.covariance = prop_cov + noise_cov
        self.state = self.model.propagate(self.state, sensor_input)

    def compute_gain(self, observation):
        """Compute the Kalman gain given the observation model."""
        obs_cov = self.model.get_measurement_noise_cov(
            self.state, self.measurement_noise)
        obs_jac = self.model.observation_jacobian(self.state, observation)
        expected_cov = Matrices.mul(
            obs_jac, self.covariance, Matrices.transpose(obs_jac))
        innovation_cov = expected_cov + obs_cov
        return Matrices.mul(self.covariance, Matrices.transpose(obs_jac),
            gs.linalg.inv(innovation_cov))

    def update(self, observation):
        """Update the current estimate given an observation."""
        innovation = self.model.innovation(self.state, observation)
        gain = self.compute_gain(observation)
        obs_jac = self.model.observation_jacobian(self.state, observation)
        cov_factor = gs.eye(self.model.dim) - Matrices.mul(gain, obs_jac)
        self.covariance = Matrices.mul(cov_factor, self.covariance)
        state_upd = Matrices.mul(gain, innovation)
        self.state = self.model.group.exp(state_upd, self.state)
        if gs.ndim(self.state) > 1:
            return gs.squeeze(self.state)


def main():
    """Carry out two examples of state estimation on groups.

     Both examples are localization problems, where only a part of the system
     is observed. The first one is a linear system, while the second one is
     non-linear.
     """
    def create_data(kalman, true_init, true_inputs, obs_freq):
        """Create data for a specific example."""
        true_traj = [1 * true_init]
        for incr in true_inputs:
            true_traj.append(kalman.model.propagate(true_traj[-1], incr))
        true_obs = [
            kalman.model.observation_model(pose)
            for pose in true_traj[obs_freq::obs_freq]]

        obs_dtype = true_obs[0].dtype
        observations = [
            np.random.multivariate_normal(obs, kalman.measurement_noise)
            for obs in true_obs]
        observations = [gs.cast(obs, obs_dtype) for obs in observations]

        input_dtype = true_inputs[0].dtype
        inputs = [gs.concatenate(
            (incr[:1], np.random.multivariate_normal(
                incr[1:], kalman.process_noise)), axis=0) for incr in true_inputs]
        inputs = [gs.cast(incr, input_dtype) for incr in inputs]

        return gs.array(true_traj), inputs, gs.array(observations)

    def estimation(observer, initial_state, inputs, obs, obs_freq):
        """Carry out the state estimation for a specific system."""
        observer.state = 1 * initial_state

        traj = [1 * observer.state]
        uncertainty = [1 * gs.diagonal(observer.covariance)]
        for i in range(n_traj):
            observer.propagate(inputs[i])
            if i > 0 and i % obs_freq == obs_freq - 1:
                observer.update(obs[(i // obs_freq)])
            traj.append(1 * observer.state)
            uncertainty.append(1 * gs.diagonal(observer.covariance))
        traj = gs.array(traj)
        uncertainty = gs.array(uncertainty)
        three_sigmas = 3 * gs.sqrt(uncertainty)

        return traj, three_sigmas

    np.random.seed(12345)
    model = LocalizationLinear()
    kalman = KalmanFilter(model)
    n_traj = 1000
    obs_freq = 50
    dt = 0.1
    init_cov = gs.array([10., 1.])
    init_cov = algebra_utils.from_vector_to_diagonal_matrix(init_cov)
    prop_cov = 0.001 * gs.eye(model.dim_noise)
    obs_cov = 10 * gs.eye(model.dim_obs)
    initial_covs = (init_cov, prop_cov, obs_cov)
    kalman.initialize_covariances(*initial_covs)

    true_state = gs.array([0., 0.])
    true_acc = gs.random.uniform(-1, 1, (n_traj, 1))
    dt_vectorized = dt * gs.ones((n_traj, 1))
    true_inputs = gs.hstack((dt_vectorized, true_acc))

    true_traj, inputs, observations = create_data(
        kalman, true_state, true_inputs, obs_freq)

    initial_state = np.random.multivariate_normal(true_state, init_cov)
    estimate, uncertainty = estimation(
        kalman, initial_state, inputs, observations, obs_freq)

    plt.figure()
    plt.plot(true_traj[:, 0], label='GT')
    plt.plot(estimate[:, 0], label='Kalman')
    plt.plot(estimate[:, 0] + uncertainty[:, 0], color='k', linestyle=':')
    plt.plot(
        estimate[:, 0] - uncertainty[:, 0], color='k', linestyle=':',
        label='3_sigma envelope')
    plt.plot(range(obs_freq, n_traj + 1, obs_freq), observations,
             marker='*', linestyle='', label='Observation')
    plt.legend()
    plt.title('1D Localization - Position')

    plt.figure()
    plt.plot(true_traj[:, 1], label='GT')
    plt.plot(estimate[:, 1], label='Kalman')
    plt.plot(estimate[:, 1] + uncertainty[:, 1], color='k', linestyle=':')
    plt.plot(
        estimate[:, 1] - uncertainty[:, 1], color='k', linestyle=':',
        label='3_sigma envelope')
    plt.legend()
    plt.title('1D Localization - Speed')

    model = Localization()
    kalman = KalmanFilter(model)
    init_cov = gs.array([1., 10., 10.])
    init_cov = algebra_utils.from_vector_to_diagonal_matrix(init_cov)
    prop_cov = 0.001 * gs.eye(model.dim_noise)
    obs_cov = 0.1 * gs.eye(model.dim_obs)
    initial_covs = (init_cov, prop_cov, obs_cov)
    kalman.initialize_covariances(*initial_covs)

    true_state = gs.zeros(model.dim)
    true_inputs = [gs.array([dt, .5, 0., 0.05]) for _ in range(n_traj)]

    true_traj, inputs, observations = create_data(
        kalman, true_state, true_inputs, obs_freq)

    initial_state = gs.array(np.random.multivariate_normal(true_state, init_cov))
    initial_state = gs.cast(initial_state, true_state.dtype)
    estimate, uncertainty = estimation(
        kalman, initial_state, inputs, observations, obs_freq)

    plt.figure()
    plt.plot(true_traj[:, 1], true_traj[:, 2], label='GT')
    plt.plot(estimate[:, 1], estimate[:, 2], label='Kalman')
    plt.scatter(observations[:, 0], observations[:, 1], s=2, c='k',
                label='Observation')
    plt.legend()
    plt.axis('equal')
    plt.title('2D Localization')

    plt.show()


if __name__ == '__main__':
    main()
