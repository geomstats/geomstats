from geomstats.test.test_case import TestCase


class LocalizationTestCase(TestCase):
    def test_propagate(self, state, sensor_input, expected, atol):
        res = self.model.propagate(state, sensor_input)
        self.assertAllClose(res, expected, atol=atol)

    def test_propagate_jacobian(self, state, sensor_input, expected, atol):
        res = self.model.propagation_jacobian(state, sensor_input)
        self.assertAllClose(res, expected, atol=atol)

    def test_observation_model(self, state, expected, atol):
        res = self.model.observation_model(state)
        self.assertAllClose(res, expected, atol=atol)

    def test_observation_jacobian(self, state, observation, expected, atol):
        res = self.model.observation_jacobian(state, observation)
        self.assertAllClose(res, expected, atol=atol)

    def test_innovation(self, state, observation, expected, atol):
        res = self.model.innovation(state, observation)
        self.assertAllClose(res, expected, atol=atol)


class NonLinearLocalizationTestCase(LocalizationTestCase):
    def test_preprocess_input(self, sensor_input, expected, atol):
        res = self.model.preprocess_input(sensor_input)
        for res_, expected_ in zip(res, expected):
            self.assertAllClose(res_, expected_, atol=atol)

    def test_rotation_matrix(self, theta, expected, atol):
        res = self.model.rotation_matrix(theta)
        self.assertAllClose(res, expected, atol=atol)

    def test_adjoint_map(self, state, expected, atol):
        res = self.model.adjoint_map(state)
        self.assertAllClose(res, expected, atol=atol)


class KalmanFilterTestCase(TestCase):
    def test_compute_gain(
        self, prior_values, process_values, obs_values, expected, atol
    ):
        self.estimator.initialize_covariances(prior_values, process_values, obs_values)

        res = self.estimator.compute_gain(None)
        self.assertAllClose(res, expected, atol=atol)

    def test_update(
        self,
        state,
        prior_values,
        process_values,
        obs_values,
        observation,
        expected_state,
        expected_cov,
        atol,
    ):
        self.estimator.state = state
        self.estimator.initialize_covariances(prior_values, process_values, obs_values)

        self.estimator.update(observation)
        self.assertAllClose(self.estimator.state, expected_state, atol=atol)
        self.assertAllClose(self.estimator.covariance, expected_cov, atol=atol)
