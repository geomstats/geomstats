"""
Unit tests for vectorization functions.
"""
import tests.helper as helper

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
            result = helper.to_vector(result)
            return result

        @geomstats.vectorization.decorator(
            ['tangent_vec_a', 'tangent_vec_b'],
            ['vector', 'vector'])
        def foo_scalar_output(tangent_vec_a, tangent_vec_b):
            result = gs.einsum(
                'ni,ni->n', tangent_vec_a, tangent_vec_b)
            result = helper.to_scalar(result)
            return result

        @geomstats.vectorization.decorator(
            ['tangent_vec_a', 'tangent_vec_b', 'in_scalar'],
            ['vector', 'vector', 'scalar'])
        def foo_scalar_input_output(tangent_vec_a, tangent_vec_b, in_scalar):
            aux = gs.einsum(
                'ni,ni->n', tangent_vec_a, tangent_vec_b)
            result = gs.einsum('n,nk->n', aux, in_scalar)
            result = helper.to_scalar(result)
            return result

        self.foo = foo
        self.foo_scalar_output = foo_scalar_output
        self.foo_scalar_input_output = foo_scalar_input_output

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

    def test_decorator_vectorization(self):
        vec_a = gs.array([[1, 2, 3], [1, 2, 3]])
        vec_b = gs.array([0, 1, 0])
        result = self.foo(vec_a, vec_b)
        expected = gs.array([[0, 2, 0], [0, 2, 0]])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)

    def test_decorator_scalar_with_squeeze_dim1(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        result = self.foo_scalar_output(vec_a, vec_b)
        expected = 2

        self.assertAllClose(result, expected)

    def test_decorator_scalar_without_squeeze_dim1(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        scalar = 4
        result = self.foo_scalar_input_output(vec_a, vec_b, scalar)
        expected = 8

        self.assertAllClose(result, expected)

    def test_decorator_scalar_output_vectorization(self):
        vec_a = gs.array([[1, 2, 3], [1, 2, 3]])
        vec_b = gs.array([0, 1, 0])
        result = self.foo_scalar_output(vec_a, vec_b)
        expected = gs.array([2, 2])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)
