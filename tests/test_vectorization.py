"""Unit tests for vectorization functions."""

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
import geomstats.vectorization


class TestVectorization(geomstats.tests.TestCase):
    def setUp(self):

        class Obj:
            def __init__(self):
                self.default_point_type = 'vector'

        @geomstats.vectorization.decorator(['vector', 'vector'])
        def func(tangent_vec_a, tangent_vec_b):
            result = gs.einsum(
                '...i,...i->...i', tangent_vec_a, tangent_vec_b)
            result = helper.to_vector(result)
            return result

        @geomstats.vectorization.decorator(['vector', 'vector'])
        def func_scalar_output(tangent_vec_a, tangent_vec_b):
            result = gs.einsum(
                '...i,...i->...', tangent_vec_a, tangent_vec_b)
            result = helper.to_scalar(result)
            return result

        @geomstats.vectorization.decorator(['vector', 'vector', 'scalar'])
        def func_scalar_input_output(tangent_vec_a, tangent_vec_b, in_scalar):
            aux = gs.einsum(
                'ni,ni->n', tangent_vec_a, tangent_vec_b)
            result = gs.einsum('n,nk->n', aux, in_scalar)
            result = helper.to_scalar(result)
            return result

        @geomstats.vectorization.decorator(['vector', 'vector', 'scalar'])
        def func_optional_input(tangent_vec_a, tangent_vec_b, in_scalar=None):
            if in_scalar is None:
                in_scalar = gs.array([[1]])
            aux = gs.einsum(
                'ni,ni->n', tangent_vec_a, tangent_vec_b)
            result = gs.einsum('n,nk->n', aux, in_scalar)
            result = helper.to_scalar(result)
            return result

        @geomstats.vectorization.decorator(
            ['else', 'vector', 'else', 'vector', 'scalar'])
        def func_else(else_a, tangent_vec_a, else_b, tangent_vec_b):
            result = (else_a + else_b) * gs.einsum(
                'ni,ni->n', tangent_vec_a, tangent_vec_b)
            result = helper.to_scalar(result)
            return result

        @geomstats.vectorization.decorator(['scalar'])
        def is_scalar_vectorized(scalar):
            is_scalar_vec = gs.ndim(scalar) == 2
            has_dim_1 = gs.shape(scalar)[-1] == 1
            result = is_scalar_vec and has_dim_1
            result = helper.to_scalar(result)
            return result

        @geomstats.vectorization.decorator(['vector'])
        def is_vector_vectorized(vector):
            is_vector_vec = gs.ndim(vector) == 2
            is_vector_vec = helper.to_scalar(is_vector_vec)
            return is_vector_vec

        @geomstats.vectorization.decorator(['matrix'])
        def is_matrix_vectorized(matrix):
            is_matrix_vec = gs.ndim(matrix) == 3
            is_matrix_vec = helper.to_scalar(is_matrix_vec)
            return is_matrix_vec

        @geomstats.vectorization.decorator(
            ['else', 'point', 'point_type'])
        def is_point_type_vector(obj, point, point_type=None):
            is_point_type_vector = point_type == 'vector'
            is_point_type_vector = helper.to_scalar(is_point_type_vector)
            return is_point_type_vector

        @geomstats.vectorization.decorator(
            ['else', 'point', 'point_type'])
        def is_point_type_matrix(obj, point, point_type=None):
            is_point_type_matrix = point_type == 'matrix'
            is_point_type_matrix = helper.to_scalar(is_point_type_matrix)
            return is_point_type_matrix

        @geomstats.vectorization.decorator(
            ['else', 'point', 'point_type'])
        def is_vector_vectorized_with_point_type(obj, point, point_type=None):
            is_vector_vec = gs.ndim(point) == 2
            is_vector_vec = helper.to_scalar(is_vector_vec)
            return is_vector_vec

        @geomstats.vectorization.decorator(
            ['else', 'point', 'point_type'])
        def is_matrix_vectorized_with_point_type(obj, point, point_type=None):
            is_matrix_vec = gs.ndim(point) == 3
            is_matrix_vec = helper.to_scalar(is_matrix_vec)
            return is_matrix_vec

        @geomstats.vectorization.decorator(
            ['else', 'point', 'point_type', 'output_point'])
        def output_1d_vector(obj, point, point_type=None):
            n_samples = point.shape[0]
            result = gs.array([[5.]] * n_samples)
            return result

        self.func = func
        self.func_scalar_output = func_scalar_output
        self.func_scalar_input_output = func_scalar_input_output
        self.func_optional_input = func_optional_input
        self.func_else = func_else
        self.is_scalar_vectorized = is_scalar_vectorized
        self.is_vector_vectorized = is_vector_vectorized
        self.is_matrix_vectorized = is_matrix_vectorized
        self.is_point_type_vector = is_point_type_vector
        self.is_point_type_matrix = is_point_type_matrix
        self.is_vector_vectorized_with_point_type = \
            is_vector_vectorized_with_point_type
        self.is_matrix_vectorized_with_point_type = \
            is_matrix_vectorized_with_point_type
        self.output_1d_vector = output_1d_vector
        self.obj = Obj()

    def test_decorator_with_squeeze_dim0(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        result = self.func(vec_a, vec_b)
        expected = gs.array([0, 2, 0])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)

    def test_decorator_with_squeeze_dim0_with_kwargs(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        result = self.func(tangent_vec_a=vec_a, tangent_vec_b=vec_b)
        expected = gs.array([0, 2, 0])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)

    def test_decorator_without_squeeze_dim0(self):
        vec_a = gs.array([[1, 2, 3]])
        vec_b = gs.array([0, 1, 0])
        result = self.func(vec_a, vec_b)
        expected = gs.array([[0, 2, 0]])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)

    def test_decorator_without_squeeze_dim0_with_kwargs(self):
        vec_a = gs.array([[1, 2, 3]])
        vec_b = gs.array([0, 1, 0])
        result = self.func(tangent_vec_a=vec_a, tangent_vec_b=vec_b)
        expected = gs.array([[0, 2, 0]])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)

    def test_decorator_vectorization(self):
        vec_a = gs.array([[1, 2, 3], [1, 2, 3]])
        vec_b = gs.array([0, 1, 0])
        result = self.func(vec_a, vec_b)
        expected = gs.array([[0, 2, 0], [0, 2, 0]])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)

    def test_decorator_vectorization_with_kwargs(self):
        vec_a = gs.array([[1, 2, 3], [1, 2, 3]])
        vec_b = gs.array([0, 1, 0])
        result = self.func(tangent_vec_a=vec_a, tangent_vec_b=vec_b)
        expected = gs.array([[0, 2, 0], [0, 2, 0]])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)

    def test_decorator_scalar_with_squeeze_dim1(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        result = self.func_scalar_output(vec_a, vec_b)
        expected = 2

        self.assertAllClose(result, expected)

    def test_decorator_scalar_with_squeeze_dim1_with_kwargs(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        result = self.func_scalar_output(
            tangent_vec_a=vec_a, tangent_vec_b=vec_b)
        expected = 2

        self.assertAllClose(result, expected)

    def test_decorator_scalar_without_squeeze_dim1(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        scalar = 4
        result = self.func_scalar_input_output(vec_a, vec_b, scalar)
        expected = 8

        self.assertAllClose(result, expected)

    def test_decorator_scalar_without_squeeze_dim1_with_kwargs(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        scalar = 4
        result = self.func_scalar_input_output(
            tangent_vec_a=vec_a, tangent_vec_b=vec_b, in_scalar=scalar)
        expected = 8

        self.assertAllClose(result, expected)

    def test_decorator_scalar_output_vectorization(self):
        vec_a = gs.array([[1, 2, 3], [1, 2, 3]])
        vec_b = gs.array([0, 1, 0])
        result = self.func_scalar_output(vec_a, vec_b)
        expected = gs.array([2, 2])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)

    def test_decorator_scalar_output_vectorization_with_kwargs(self):
        vec_a = gs.array([[1, 2, 3], [1, 2, 3]])
        vec_b = gs.array([0, 1, 0])
        result = self.func_scalar_output(
            tangent_vec_a=vec_a, tangent_vec_b=vec_b)
        expected = gs.array([2, 2])

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)

    def test_decorator_optional_input(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        result = self.func_optional_input(vec_a, vec_b)
        expected = 2

        self.assertAllClose(result, expected)

    def test_decorator_optional_input_with_kwargs(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        result = self.func_optional_input(
            tangent_vec_a=vec_a, tangent_vec_b=vec_b, in_scalar=3)
        expected = 6

        self.assertAllClose(result, expected)

    def test_decorator_optional_input_with_optional_kwargs(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        result = self.func_optional_input(
            tangent_vec_a=vec_a, tangent_vec_b=vec_b)
        expected = 2

        self.assertAllClose(result, expected)

    def test_decorator_else(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        else_a = 1
        else_b = 1
        result = self.func_else(else_a, vec_a, else_b, vec_b)
        expected = 4

        self.assertAllClose(result, expected)

    def test_decorator_else_with_kwargs(self):
        vec_a = gs.array([1, 2, 3])
        vec_b = gs.array([0, 1, 0])
        else_a = 1
        else_b = 1
        result = self.func_else(
            else_a=else_a, tangent_vec_a=vec_a,
            else_b=else_b, tangent_vec_b=vec_b)
        expected = 4

        self.assertAllClose(result, expected)

    def test_is_scalar_vectorized(self):
        scalar = 1.3
        result = self.is_scalar_vectorized(scalar)
        expected = True
        self.assertAllClose(result, expected)

    def test_is_scalar_vectorized_with_kwargs(self):
        scalar = 1.3
        result = self.is_scalar_vectorized(scalar=scalar)
        expected = True
        self.assertAllClose(result, expected)

    def test_is_vector_vectorized(self):
        vector = gs.array([1.3, 3.3])
        result = self.is_vector_vectorized(vector)
        expected = True
        self.assertAllClose(result, expected)

    def test_is_vector_vectorizedi_with_kwargs(self):
        vector = gs.array([1.3, 3.3])
        result = self.is_vector_vectorized(vector=vector)
        expected = True
        self.assertAllClose(result, expected)

    def test_is_matrix_vectorized(self):
        matrix = gs.array([[1.3, 3.3], [1.2, 3.1]])
        result = self.is_matrix_vectorized(matrix)
        expected = True
        self.assertAllClose(result, expected)

    def test_is_matrix_vectorized_with_kwargs(self):
        matrix = gs.array([[1.3, 3.3], [1.2, 3.1]])
        result = self.is_matrix_vectorized(matrix=matrix)
        expected = True
        self.assertAllClose(result, expected)

    def test_vectorize_args(self):
        point_types = ['scalar']
        args = (1.3,)
        result = geomstats.vectorization.vectorize_args(point_types, args)
        expected = (gs.array([[1.3]]),)
        self.assertAllClose(result, expected)

    def test_vectorize_kwargs(self):
        point_types = ['scalar']
        kwargs = {'scalar_name': 1.3}
        result_dict = geomstats.vectorization.vectorize_kwargs(
            point_types, kwargs)
        expected_dict = {'scalar_name': gs.array([[1.3]])}

        keys = expected_dict.keys()
        result = gs.array([result_dict[key] for key in keys])
        expected = gs.array([expected_dict[key] for key in keys])

        self.assertAllClose(result, expected)

    def test_is_point_type_vector(self):
        point = gs.array([1., 2., 3.])
        result = self.is_point_type_vector(
            self.obj, point, point_type='vector')
        expected = True
        self.assertAllClose(result, expected)

    def test_is_point_type_vector_optional(self):
        point = gs.array([1., 2., 3.])
        result = self.is_point_type_vector(
            self.obj, point)
        expected = True
        self.assertAllClose(result, expected)

    def test_is_point_type_matrix(self):
        point = gs.array([[1., 2., 3.], [2., 3., 4.]])
        result = self.is_point_type_matrix(
            self.obj, point, point_type='matrix')
        expected = True
        self.assertAllClose(result, expected)

    def test_is_vector_vectorized_with_point_type(self):
        vector = gs.array([1.3, 3.3])
        result = self.is_vector_vectorized_with_point_type(
            self.obj, point=vector, point_type='vector')
        expected = True
        self.assertAllClose(result, expected)

    def test_is_vector_vectorized_with_optional_point_type(self):
        vector = gs.array([1.3, 3.3])
        result = self.is_vector_vectorized_with_point_type(
            self.obj, point=vector)
        expected = True
        self.assertAllClose(result, expected)

    def test_is_matrix_vectorized_with_point_type(self):
        matrix = gs.array([[1.3, 3.3], [1.2, 3.1]])
        result = self.is_matrix_vectorized_with_point_type(
            self.obj, matrix, point_type='matrix')
        expected = True
        self.assertAllClose(result, expected)

    def test_output_single_1d_vector(self):
        point = gs.array([1., 2., 3.])
        result = self.output_1d_vector(
            self.obj, point)
        expected = gs.array([5.])
        self.assertAllClose(result, expected)

    def test_output_multiple_1d_vector(self):
        point = gs.array([[1., 2., 3.], [2., 3., 4.]])
        result = self.output_1d_vector(
            self.obj, point)
        expected = gs.array([[5.], [5.]])
        self.assertAllClose(result, expected)
