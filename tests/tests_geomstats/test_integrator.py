"""Test for the integrators."""

import pytest

import geomstats.backend as gs
import geomstats.integrator as integrator
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.test.test_case import TestCase


class TestIntegrator(TestCase):
    def setup_method(self):
        self.dimension = 4
        self.dt = 0.1
        self.euclidean = Euclidean(self.dimension)
        self.matrices = Matrices(self.dimension, self.dimension)
        self.intercept = self.euclidean.random_point()
        self.slope = Matrices.to_symmetric(self.matrices.random_point())

    @staticmethod
    def function_linear(_state, _time):
        return 2.0

    def _test_step(self, step):
        state = self.intercept
        result = step(self.function_linear, state, 0.0, self.dt)
        expected = state + 2 * self.dt

        self.assertAllClose(result, expected)

    def test_symplectic_euler_step(self):
        with pytest.raises(NotImplementedError):
            self._test_step(integrator.symplectic_euler_step)

    def test_leapfrog_step(self):
        initial_position = gs.array([0.0, -1.0, 0.0])
        initial_velocity = gs.array([1.0, 0.0, 0.0])
        initial_state = gs.array([initial_position, initial_velocity])
        time = 0.0
        dt = 0.1

        def harmonic_force(state, _time):
            return -state

        new_state = integrator.leapfrog_step(harmonic_force, initial_state, time, dt)

        expected_position = gs.array([gs.sin(dt), -gs.cos(dt), 0.0]).reshape(1, 3)
        expected_velocity = gs.array([gs.cos(dt), gs.sin(dt), 0.0]).reshape(1, 3)
        expected_state = gs.concatenate([expected_position, expected_velocity], axis=-2)

        self.assertAllClose(new_state, expected_state, rtol=1e-3, atol=1e-3)

    def test_euler_step(self):
        self._test_step(integrator.euler_step)

    def test_rk2_step(self):
        self._test_step(integrator.rk2_step)

    def test_rk4_step(self):
        self._test_step(integrator.rk4_step)

    def test_integrator(self):
        initial_state = self.euclidean.random_point(2)

        def position_velocity_function(state, _time):
            _, velocity = state
            return gs.stack([velocity, gs.zeros_like(velocity)])

        for step in ["euler", "rk2", "rk4"]:
            flow = integrator.integrate(
                position_velocity_function, initial_state, step=step
            )
            result = flow[-1][0]
            expected = initial_state[0] + initial_state[1]

            self.assertAllClose(result, expected)

        def position_function(position, _time):
            return 0

        for step in ["leapfrog"]:
            flow = integrator.integrate(position_function, initial_state, step=step)
            result = flow[-1][..., 0, :]
            expected = initial_state[0] + initial_state[1]

            self.assertAllClose(result, expected)
