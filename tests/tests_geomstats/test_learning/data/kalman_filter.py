import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.test.data import TestData


class LocalizationLinearTestData(TestData):
    def propagate_test_data(self):
        time_step = 0.5
        acc = 2.0
        data = [
            dict(
                state=gs.array([0.5, 1.0]),
                sensor_input=gs.array([time_step, acc]),
                expected=gs.array([1.0, 2.0]),
            )
        ]

        return self.generate_tests(data)

    def propagate_jacobian_test_data(self):
        time_step = 0.5
        acc = 2.0
        data = [
            dict(
                state=None,
                sensor_input=gs.array([time_step, acc]),
                expected=gs.array([[1.0, 0.5], [0.0, 1.0]]),
            )
        ]

        return self.generate_tests(data)

    def observation_model_test_data(self):
        data = [
            dict(
                state=gs.array([0.5, 1.0]),
                expected=gs.array([0.5]),
            )
        ]
        return self.generate_tests(data)

    def observation_jacobian_test_data(self):
        data = [
            dict(
                state=None,
                observation=None,
                expected=gs.array([[1.0, 0.0]]),
            )
        ]
        return self.generate_tests(data)

    def innovation_test_data(self):
        data = [
            dict(
                state=gs.array([0.5, 1.0]),
                observation=gs.array([0.7]),
                expected=gs.array([0.2]),
            )
        ]
        return self.generate_tests(data)


class LocalizationTestData(TestData):
    def propagate_test_data(self):
        time_step = gs.array([0.5])
        linear_vel = gs.array([1.0, 0.5])
        angular_vel = gs.array([0.0])

        initial_state = gs.array([0.5, 1.0, 2.0])
        angle = initial_state[0]
        rotation = gs.array(
            [[gs.cos(angle), -gs.sin(angle)], [gs.sin(angle), gs.cos(angle)]]
        )

        next_position = initial_state[1:] + time_step * gs.einsum("...ij,...j->...i", rotation, linear_vel)
        expected = gs.concatenate((gs.array([angle]), next_position), axis=0)

        data = [
            dict(
                state=initial_state,
                sensor_input=gs.concatenate(
                    (time_step, linear_vel, angular_vel), axis=0
                ),
                expected=expected,
            )
        ]

        return self.generate_tests(data)

    def propagate_jacobian_test_data(self):
        time_step = gs.array([0.5])
        linear_vel = gs.array([1.0, 0.5])
        angular_vel = gs.array([0.0])

        first_line = gs.eye(1, 3)
        last_lines = gs.hstack((gs.array([[-0.25], [0.5]]), gs.eye(2)))
        expected = gs.vstack((first_line, last_lines))

        data = [
            dict(
                state=None,
                sensor_input=gs.concatenate(
                    (time_step, linear_vel, angular_vel), axis=0
                ),
                expected=expected,
            )
        ]

        return self.generate_tests(data)

    def observation_model_test_data(self):
        data = [
            dict(
                state=gs.array([0.5, 1.0, 2.0]),
                expected=gs.array([1.0, 2.0]),
            )
        ]
        return self.generate_tests(data)

    def observation_jacobian_test_data(self):
        data = [
            dict(
                state=None,
                observation=None,
                expected=gs.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            )
        ]
        return self.generate_tests(data)

    def innovation_test_data(self):
        initial_state = gs.array([0.5, 1.0, 2.0])

        angle = initial_state[0]
        rotation = gs.array(
            [[gs.cos(angle), -gs.sin(angle)], [gs.sin(angle), gs.cos(angle)]]
        )
        expected = gs.einsum("...ij,...j->...i", gs.transpose(rotation), gs.array([-0.3, 0.1]))

        data = [
            dict(
                state=initial_state,
                observation=gs.array([0.7, 2.1]),
                expected=expected,
            )
        ]
        return self.generate_tests(data)

    def preprocess_input_test_data(self):
        time_step = gs.array([0.5])
        linear_vel = gs.array([1.0, 0.5])
        angular_vel = gs.array([0.0])

        data = [
            dict(
                sensor_input=gs.concatenate(
                    (time_step, linear_vel, angular_vel), axis=0
                ),
                expected=(time_step[0], linear_vel, angular_vel),
            )
        ]

        return self.generate_tests(data)

    def rotation_matrix_test_data(self):
        angle = 0.5
        data = [
            dict(
                theta=angle,
                expected=gs.array(
                    [[gs.cos(angle), -gs.sin(angle)], [gs.sin(angle), gs.cos(angle)]]
                ),
            )
        ]
        return self.generate_tests(data)

    def adjoint_map_test_data(self):
        initial_state = gs.array([0.5, 1.0, 2.0])
        angle = initial_state[0]
        rotation = gs.array(
            [[gs.cos(angle), -gs.sin(angle)], [gs.sin(angle), gs.cos(angle)]]
        )
        first_line = gs.eye(1, 3)
        last_lines = gs.hstack((gs.array([[2.0], [-1.0]]), rotation))
        expected = gs.vstack((first_line, last_lines))

        data = [
            dict(
                state=initial_state,
                expected=expected,
            )
        ]

        return self.generate_tests(data)


class KalmanFilterTestData(TestData):
    def compute_gain_test_data(self):
        innovation_cov = 3 * gs.eye(1)
        data = [
            dict(
                prior_values=gs.eye(2),
                process_values=gs.eye(1),
                obs_values=2.0 * gs.eye(1),
                expected=gs.vstack(
                    (1.0 / innovation_cov, gs.zeros_like(innovation_cov))
                ),
            )
        ]

        return self.generate_tests(data)

    def update_test_data(self):
        data = [
            dict(
                state=gs.zeros(2),
                prior_values=gs.eye(2),
                process_values=gs.eye(1),
                obs_values=2.0 * gs.eye(1),
                observation=gs.array([0.6]),
                expected_state=gs.array([0.2, 0.0]),
                expected_cov=from_vector_to_diagonal_matrix(gs.array([2.0 / 3.0, 1.0])),
            )
        ]
        return self.generate_tests(data)
