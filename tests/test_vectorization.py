"""
Unit tests for vectorization functions.
"""

import geomstats.backend as gs
import geomstats.tests
import geomstats.vectorization


class TestVectorizationMethods(geomstats.tests.TestCase):
    def setUp(self):
        @geomstats.vectorization.decorator(
                ['tangent_vec_a', 'tangent_vec_b'],
                ['vector', 'vector'])
        def foo(tangent_vec_a, tangent_vec_b):
            result = gs.einsum(
                'ni,ni->ni', tangent_vec_a, tangent_vec_b)
            return result
        self.foo = foo

    def test_decorator_with_squeeze_dim0(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        result = self.foo(vec_a, vec_b)
        expected = gs.array([0, 2, 0])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)

    def test_decorator_without_squeeze_dim0(self):
        vec_a = gs.array([[1, 2, 3]])
        vec_b = gs.array([0, 1, 0])
        result = self.foo(vec_a, vec_b)
        expected = gs.array([[0, 2, 0]])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)

    def test_decorator_with_n_samples(self):
        vec_a = gs.array([[1, 2, 3], [1, 2, 3]])
        vec_b = gs.array([0, 1, 0])
        result = self.foo(vec_a, vec_b)
        expected = gs.array([[0, 2, 0], [0, 2, 0]])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)
