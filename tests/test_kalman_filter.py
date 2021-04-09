"""Unit tests for Kalman filter."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.matrices import Matrices
from geomstats.learning.kalman_filter import KalmanFilter
from geomstats.learning.kalman_filter import Localization
from geomstats.learning.kalman_filter import LocalizationLinear


class TestKalmanFilter(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(123)
        self.linear_model = LocalizationLinear()
        self.nonlinear_model = Localization()
        self.kalman = KalmanFilter(self.linear_model)
        self.prior_cov = gs.eye(2)
        self.process_cov = gs.eye(1)
        self.obs_cov = 2. * gs.eye(1)

    def test_LocalizationLinear_propagate(self):
        initial_state = gs.array([0.5, 1.])
        time_step = 0.5
        acc = 2.
        increment = gs.array([time_step, acc])

        expected = gs.array([1., 2.])
        result = self.linear_model.propagate(initial_state, increment)
        self.assertAllClose(expected, result)

    def test_LocalizationLinear_propagation_jacobian(self):
        time_step = 0.5
        acc = 2.
        increment = gs.array([time_step, acc])
        expected = gs.array([[1., 0.5],
                             [0., 1.]])
        result = self.linear_model.propagation_jacobian(None, increment)
        self.assertAllClose(expected, result)

    def test_LocalizationLinear_observation_model(self):
        initial_state = gs.array([0.5, 1.])
        expected = gs.array([0.5])
        result = self.linear_model.observation_model(initial_state)
        self.assertAllClose(expected, result)

    def test_LocalizationLinear_observation_jacobian(self):
        expected = gs.array([[1., 0.]])
        result = self.linear_model.observation_jacobian(None, None)
        self.assertAllClose(expected, result)

    def test_LocalizationLinear_innovation(self):
        initial_state = gs.array([0.5, 1.])
        measurement = gs.array([0.7])
        expected = gs.array([0.2])
        result = self.linear_model.innovation(initial_state, measurement)
        self.assertAllClose(expected, result)

    def test_Localization_preprocess_input(self):
        time_step = gs.array([0.5])
        linear_vel = gs.array([1., 0.5])
        angular_vel = gs.array([0.])
        increment = gs.concatenate((
            time_step, linear_vel, angular_vel), axis=0)

        expected = time_step[0], linear_vel, angular_vel
        result = self.nonlinear_model.preprocess_input(increment)
        for i in range(3):
            self.assertAllClose(expected[i], result[i])

    def test_Localization_rotation_matrix(self):
        initial_state = gs.array([0.5, 1., 2.])

        angle = initial_state[0]
        rotation = gs.array([[gs.cos(angle), -gs.sin(angle)],
                             [gs.sin(angle), gs.cos(angle)]])
        expected = rotation
        result = self.nonlinear_model.rotation_matrix(angle)
        self.assertAllClose(expected, result)

    def test_Localization_adjoint_map(self):
        initial_state = gs.array([0.5, 1., 2.])

        angle = initial_state[0]
        rotation = gs.array([[gs.cos(angle), -gs.sin(angle)],
                             [gs.sin(angle), gs.cos(angle)]])
        first_line = gs.eye(1, 3)
        last_lines = gs.hstack((gs.array([[2.], [-1.]]), rotation))
        expected = gs.vstack((first_line, last_lines))
        result = self.nonlinear_model.adjoint_map(initial_state)
        self.assertAllClose(expected, result)

    def test_Localization_propagate(self):
        initial_state = gs.array([0.5, 1., 2.])
        time_step = gs.array([0.5])
        linear_vel = gs.array([1., 0.5])
        angular_vel = gs.array([0.])
        increment = gs.concatenate((
            time_step, linear_vel, angular_vel), axis=0)

        angle = initial_state[0]
        rotation = gs.array([[gs.cos(angle), -gs.sin(angle)],
                             [gs.sin(angle), gs.cos(angle)]])
        next_position = initial_state[1:] + time_step * gs.matmul(
            rotation, linear_vel)
        expected = gs.concatenate((gs.array([angle]), next_position), axis=0)
        result = self.nonlinear_model.propagate(initial_state, increment)
        self.assertAllClose(expected, result)

    def test_Localization_propagation_jacobian(self):
        time_step = gs.array([0.5])
        linear_vel = gs.array([1., 0.5])
        angular_vel = gs.array([0.])
        increment = gs.concatenate((
            time_step, linear_vel, angular_vel), axis=0)

        first_line = gs.eye(1, 3)
        last_lines = gs.hstack((gs.array([[-0.25], [0.5]]), gs.eye(2)))
        expected = gs.vstack((first_line, last_lines))
        result = self.nonlinear_model.propagation_jacobian(None, increment)
        self.assertAllClose(expected, result)

    def test_Localization_observation_model(self):
        initial_state = gs.array([0.5, 1., 2.])
        expected = gs.array([1., 2.])
        result = self.nonlinear_model.observation_model(initial_state)
        self.assertAllClose(expected, result)

    def test_Localization_observation_jacobian(self):
        expected = gs.array([[0., 1., 0.],
                             [0., 0., 1.]])
        result = self.nonlinear_model.observation_jacobian(None, None)
        self.assertAllClose(expected, result)

    def test_Localization_innovation(self):
        initial_state = gs.array([0.5, 1., 2.])
        measurement = gs.array([0.7, 2.1])

        angle = initial_state[0]
        rotation = gs.array([[gs.cos(angle), -gs.sin(angle)],
                             [gs.sin(angle), gs.cos(angle)]])
        expected = gs.matmul(gs.transpose(rotation), gs.array([-0.3, 0.1]))
        result = self.nonlinear_model.innovation(initial_state, measurement)
        self.assertAllClose(expected, result)

    def test_initialize_covariances(self):
        self.kalman.initialize_covariances(
            self.prior_cov, self.process_cov, self.obs_cov)
        self.assertAllClose(self.kalman.covariance, self.prior_cov)
        self.assertAllClose(self.kalman.process_noise, self.process_cov)
        self.assertAllClose(self.kalman.measurement_noise, self.obs_cov)

    def test_propagate(self):
        self.kalman.initialize_covariances(
            self.prior_cov, self.process_cov, self.obs_cov)
        time_step = 0.5
        acc = 2.
        increment = gs.array([time_step, acc])
        state_jacobian = self.linear_model.propagation_jacobian(
            self.kalman.state, increment)
        noise_jacobian = self.linear_model.noise_jacobian(
            self.kalman.state, increment)
        expected_covariance = Matrices.mul(
            state_jacobian,
            self.kalman.covariance,
            gs.transpose(state_jacobian)) \
            + Matrices.mul(
            noise_jacobian,
            self.kalman.process_noise,
            gs.transpose(noise_jacobian))
        expected_state = self.linear_model.propagate(
            self.kalman.state, increment)
        self.kalman.propagate(increment)
        self.assertAllClose(self.kalman.state, expected_state)
        self.assertAllClose(self.kalman.covariance, expected_covariance)

    def test_compute_gain(self):
        self.kalman.initialize_covariances(
            self.prior_cov, self.process_cov, self.obs_cov)
        innovation_cov = 3 * gs.eye(1)
        expected = gs.vstack(
            (1. / innovation_cov, gs.zeros_like(innovation_cov)))
        result = self.kalman.compute_gain(None)
        self.assertAllClose(expected, result)

    def test_update(self):
        self.kalman.state = gs.zeros(2)
        self.kalman.initialize_covariances(
            self.prior_cov, self.process_cov, self.obs_cov)
        measurement = gs.array([0.6])
        expected_cov = from_vector_to_diagonal_matrix(gs.array([2. / 3., 1.]))
        expected_state = gs.array([0.2, 0.])
        self.kalman.update(measurement)
        self.assertAllClose(expected_state, self.kalman.state)
        self.assertAllClose(expected_cov, self.kalman.covariance)
