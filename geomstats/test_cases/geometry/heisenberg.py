import pytest

import geomstats.backend as gs
from geomstats.test.random import VectorSpaceRandomDataGenerator
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import VectorSpaceTestCase
from geomstats.test_cases.geometry.lie_group import LieGroupTestCase


class HeisenbergVectorsRandomDataGenerator(VectorSpaceRandomDataGenerator):
    def random_upper_triangular_matrix(self, n_points=1):
        if n_points == 1:
            size = 3
            expected_shape = (3, 3)
            indices = [(0, 1), (0, 2), (1, 2)]
        else:
            size = (n_points, 3)
            expected_shape = (n_points, 3, 3)
            indices = []
            for k in range(n_points):
                indices.extend([(k, 0, 1), (k, 0, 2), (k, 1, 2)])

        vec = gs.random.uniform(size=size)
        return gs.array_from_sparse(indices, vec, expected_shape) + gs.eye(3)


class HeisenbergVectorsTestCase(LieGroupTestCase, VectorSpaceTestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = HeisenbergVectorsRandomDataGenerator(self.space)

    def test_upper_triangular_matrix_from_vector(self, point, expected, atol):
        res = self.space.upper_triangular_matrix_from_vector(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_upper_triangular_matrix_from_vector_vec(self, n_reps, atol):
        point = self.data_generator.random_point()

        expected = self.space.upper_triangular_matrix_from_vector(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_vector_from_upper_triangular_matrix(self, matrix, expected, atol):
        res = self.space.vector_from_upper_triangular_matrix(matrix)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_vector_from_upper_triangular_matrix_vec(self, n_reps, atol):
        matrix = self.data_generator.random_upper_triangular_matrix()
        expected = self.space.vector_from_upper_triangular_matrix(matrix)

        vec_data = generate_vectorization_data(
            data=[dict(matrix=matrix, expected=expected, atol=atol)],
            arg_names=["matrix"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_vector_from_upper_triangular_matrix_after_upper_triangular_matrix_from_vector(
        self, n_points, atol
    ):
        vector = self.data_generator.random_point(n_points)
        mat = self.space.upper_triangular_matrix_from_vector(vector)
        vector_ = self.space.vector_from_upper_triangular_matrix(mat)
        self.assertAllClose(vector_, vector, atol=atol)

    @pytest.mark.random
    def test_upper_triangular_matrix_from_vector_after_vector_from_upper_triangular_matrix(
        self, n_points, atol
    ):
        mat = self.data_generator.random_upper_triangular_matrix(n_points)
        vec = self.space.vector_from_upper_triangular_matrix(mat)
        mat_ = self.space.upper_triangular_matrix_from_vector(vec)
        self.assertAllClose(mat_, mat, atol=atol)
