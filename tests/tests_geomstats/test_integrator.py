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
        with pytest.raises(NotImplementedError):
            self._test_step(integrator.leapfrog_step)

    def test_euler_step(self):
        self._test_step(integrator.euler_step)

    def test_rk2_step(self):
        self._test_step(integrator.rk2_step)

    def test_rk4_step(self):
        self._test_step(integrator.rk4_step)

    def test_integrator(self):
        initial_state = self.euclidean.random_point(2)

        def function(state, _time):
            _, velocity = state
            return gs.stack([velocity, gs.zeros_like(velocity)])

        for step in ["euler", "rk2", "rk4"]:
            flow = integrator.integrate(function, initial_state, step=step)
            result = flow[-1][0]
            expected = initial_state[0] + initial_state[1]

            self.assertAllClose(result, expected)
