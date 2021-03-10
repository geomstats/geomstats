"""Test for the integrators."""

import geomstats.backend as gs
import geomstats.integrator as integrator
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices


class TestIntegrator(geomstats.tests.TestCase):
    def setUp(self):
        self.dimension = 4
        self.dt = 0.1
        self.euclidean = Euclidean(self.dimension)
        self.matrices = Matrices(self.dimension, self.dimension)
        self.intercept = self.euclidean.random_uniform(1)
        self.slope = Matrices.to_symmetric(self.matrices.random_uniform(1))

    def function_linear(self, point, vector):
        return point, - gs.dot(self.slope, vector)

    def test_euler_step(self):
        state = (self.intercept, self.slope)
        result = len(
            integrator.euler_step(state, self.function_linear, self.dt))
        expected = len(state)

        self.assertAllClose(result, expected)

    def test_rk4_step(self):
        state = (self.intercept, self.slope)
        result = len(integrator.rk4_step(
            state, self.function_linear, self.dt))
        expected = len(state)

        self.assertAllClose(result, expected)

    def test_integrator(self):
        initial_state = self.euclidean.random_uniform(2)

        def function(_, velocity):
            return velocity, gs.zeros_like(velocity)

        for step in ['euler', 'rk4']:
            flow, _ = integrator.integrate(function, initial_state, step=step)
            result = flow[-1]
            expected = initial_state[0] + initial_state[1]

            self.assertAllClose(result, expected)
