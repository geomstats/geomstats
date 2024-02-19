import pytest

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import MatrixVectorSpaceTestCase
from geomstats.test_cases.geometry.flat_riemannian_metric import (
    FlatRiemannianMetricTestCase,
)


class MatrixOperationsTestCase(TestCase):
    @staticmethod
    def _get_transpose_shape(mat):
        return mat.shape[::-1][:2]

    def test_equal(self, mat_a, mat_b, expected, atol):
        res = Matrices.equal(mat_a, mat_b, atol=atol)
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_equal_vec(self, n_reps, atol):
        mat_a = self.data_generator.random_mat()
        mat_b = self.data_generator.random_mat(shape=mat_a.shape[-2:])

        expected = Matrices.equal(mat_a, mat_b)

        vec_data = generate_vectorization_data(
            data=[dict(mat_a=mat_a, mat_b=mat_b, expected=expected, atol=atol)],
            arg_names=["mat_a", "mat_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_equal_true(self, n_points, atol):
        mat_a = mat_b = self.data_generator.random_mat(n_points)

        self.test_equal(mat_a, mat_b, True, atol=atol)

    @pytest.mark.random
    def test_equal_false(self, n_points, atol):
        mat_a = self.data_generator.random_mat(n_points)
        mat_b = self.data_generator.random_mat(n_points, shape=mat_a.shape[-2:])

        self.test_equal(mat_a, mat_b, False, atol=atol)

    def test_mul_reduce(self, mats, expected, atol):
        res = Matrices.mul(*mats)
        self.assertAllClose(res, expected, atol=atol)

    def test_mul(self, mat_a, mat_b, expected, atol):
        res = Matrices.mul(mat_a, mat_b)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_mul_vec(self, n_reps, atol):
        mat_a = self.data_generator.random_mat()
        transpose_shape = self._get_transpose_shape(mat_a)
        mat_b = self.data_generator.random_mat(shape=transpose_shape)

        expected = Matrices.mul(mat_a, mat_b)

        vec_data = generate_vectorization_data(
            data=[dict(mat_a=mat_a, mat_b=mat_b, expected=expected, atol=atol)],
            arg_names=["mat_a", "mat_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_mul_identity(self, n_points, atol):
        mat_a = self.data_generator.random_mat(n_points)
        identity = gs.eye(mat_a.shape[-1])

        self.test_mul(mat_a, identity, mat_a, atol)

    def test_bracket(self, mat_a, mat_b, expected, atol):
        res = Matrices.bracket(mat_a, mat_b)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_bracket_vec(self, n_reps, atol):
        mat_a = self.data_generator.random_square_mat()
        mat_b = self.data_generator.random_square_mat(shape=mat_a.shape[-2:])

        expected = Matrices.bracket(mat_a, mat_b)

        vec_data = generate_vectorization_data(
            data=[dict(mat_a=mat_a, mat_b=mat_b, expected=expected, atol=atol)],
            arg_names=["mat_a", "mat_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_bracket_bilinearity(self, n_points, atol):
        mat_a = self.data_generator.random_square_mat(n_points)
        shape = mat_a.shape[-2:]
        mat_b = self.data_generator.random_square_mat(shape=shape)
        mat_c = self.data_generator.random_square_mat(shape=shape)

        a, b = gs.random.uniform(size=(2,))

        mat_ab = a * mat_a + b * mat_b
        left = Matrices.bracket(mat_ab, mat_c)
        right = a * Matrices.bracket(mat_a, mat_c) + b * Matrices.bracket(mat_b, mat_c)
        self.assertAllClose(left, right, atol=atol)

        left = Matrices.bracket(mat_c, mat_ab)
        right = a * Matrices.bracket(mat_c, mat_a) + b * Matrices.bracket(mat_c, mat_b)
        self.assertAllClose(left, right, atol=atol)

    def test_transpose(self, mat, expected, atol):
        res = Matrices.transpose(mat)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_transpose_vec(self, n_reps, atol):
        mat = self.data_generator.random_mat()
        expected = Matrices.transpose(mat)

        vec_data = generate_vectorization_data(
            data=[dict(mat=mat, expected=expected, atol=atol)],
            arg_names=["mat"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_transpose_sym(self, n_points, atol):
        mat = self.data_generator.random_symmetric_mat(n_points)

        self.test_transpose(mat, mat, atol)

    def test_diagonal(self, mat, expected, atol):
        res = Matrices.diagonal(mat)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_diagonal_vec(self, n_reps, atol):
        mat = self.data_generator.random_mat()

        expected = Matrices.diagonal(mat)
        vec_data = generate_vectorization_data(
            data=[dict(mat=mat, expected=expected, atol=atol)],
            arg_names=["mat"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_diagonal_sum(self, n_points, atol):
        mat = self.data_generator.random_square_mat(n_points)

        diag = Matrices.diagonal(mat)

        sum_diag = gs.sum(diag, axis=-1)
        trace = gs.trace(mat)
        self.assertAllClose(sum_diag, trace, atol=atol)

    def test_is_square(self, mat, expected):
        res = Matrices.is_square(mat)
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_is_square_vec(self, n_reps):
        mat = self.data_generator.random_square_mat()

        expected = Matrices.is_square(mat)
        vec_data = generate_vectorization_data(
            data=[dict(mat=mat, expected=expected)],
            arg_names=["mat"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_is_square_true(self, n_points):
        mat = self.data_generator.random_square_mat(n_points)
        self.test_is_square(mat, True)

    @pytest.mark.random
    def test_is_square_false(self, n_points):
        mat = self.data_generator.random_non_square_mat(n_points)
        self.test_is_square(mat, False)

    def test_is_property(self, property_name, mat, expected, atol):
        func = getattr(Matrices, f"is_{property_name}")
        res = func(mat, atol=atol)
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_is_property_vec(self, property_name, n_reps, atol):
        mat = getattr(self.data_generator, f"random_{property_name}_mat")()

        func = getattr(Matrices, f"is_{property_name}")
        expected = func(mat, atol=atol)

        vec_data = generate_vectorization_data(
            data=[
                dict(property_name=property_name, mat=mat, expected=expected, atol=atol)
            ],
            arg_names=["mat"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_is_property_true(self, property_name, n_points, atol):
        mat = getattr(self.data_generator, f"random_{property_name}_mat")(n_points)
        self.test_is_property(property_name, mat, True, atol)

    @pytest.mark.random
    def test_is_property_false_square(self, property_name, n_points, atol):
        mat = self.data_generator.random_square_mat(n_points)
        self.test_is_property(property_name, mat, False, atol)

    @pytest.mark.random
    def test_to_property_is_property(self, property_name, n_points, atol):
        square_mat = self.data_generator.random_square_mat(n_points)

        to_func = getattr(Matrices, f"to_{property_name}")
        is_func = getattr(Matrices, f"is_{property_name}")

        mat = to_func(square_mat)
        res = is_func(mat, atol=atol)

        self.assertAllEqual(res, True)

    @pytest.mark.random
    def test_to_lower_triangular_diagonal_scaled_is_lower_triangular(
        self, n_points, atol
    ):
        square_mat = self.data_generator.random_square_mat(n_points)

        mat = Matrices.to_lower_triangular_diagonal_scaled(square_mat)
        res = Matrices.is_lower_triangular(mat, atol=atol)

        self.assertAllEqual(res, True)

    def test_congruent(self, mat_1, mat_2, expected, atol):
        res = Matrices.congruent(
            mat_1,
            mat_2,
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_congruent_vec(self, n_reps, atol):
        mat_1 = self.data_generator.random_square_mat()
        shape = mat_1.shape[-2:]
        mat_2 = self.data_generator.random_square_mat(shape=shape)

        expected = Matrices.congruent(mat_1, mat_2)

        vec_data = generate_vectorization_data(
            data=[dict(mat_1=mat_1, mat_2=mat_2, expected=expected, atol=atol)],
            arg_names=["mat_1", "mat_2"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_frobenius_product(self, mat_1, mat_2, expected, atol):
        res = Matrices.frobenius_product(
            mat_1,
            mat_2,
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_frobenius_product_vec(self, n_reps, atol):
        mat_1 = self.data_generator.random_mat()
        shape = mat_1.shape[-2:]
        mat_2 = self.data_generator.random_mat(shape=shape)

        expected = Matrices.frobenius_product(mat_1, mat_2)

        vec_data = generate_vectorization_data(
            data=[dict(mat_1=mat_1, mat_2=mat_2, expected=expected, atol=atol)],
            arg_names=["mat_1", "mat_2"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_trace_product(self, mat_1, mat_2, expected, atol):
        res = Matrices.trace_product(
            mat_1,
            mat_2,
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_trace_product_vec(self, n_reps, atol):
        mat_1 = self.data_generator.random_mat()
        shape = mat_1.shape[-2:]
        mat_2 = gs.transpose(self.data_generator.random_mat(shape=shape))

        expected = Matrices.trace_product(mat_1, mat_2)

        vec_data = generate_vectorization_data(
            data=[dict(mat_1=mat_1, mat_2=mat_2, expected=expected, atol=atol)],
            arg_names=["mat_1", "mat_2"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_flatten(self, mat, expected, atol):
        res = Matrices.flatten(mat)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_flatten_vec(self, n_reps, atol):
        mat = self.data_generator.random_mat()

        expected = Matrices.flatten(mat)

        vec_data = generate_vectorization_data(
            data=[dict(mat=mat, expected=expected, atol=atol)],
            arg_names=["mat"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_align_matrices(self, point, base_point, expected, atol):
        res = Matrices.align_matrices(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_align_matrices_vec(self, n_reps, atol):
        point = self.data_generator.random_mat()
        shape = point.shape[-2:]
        base_point = self.data_generator.random_mat(shape=shape)

        expected = Matrices.align_matrices(point, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(point=point, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["point", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)


class MatricesTestCase(MatrixVectorSpaceTestCase):
    @pytest.mark.random
    def test_reshape_after_flatten(self, n_points, atol):
        point = self.data_generator.random_point(n_points)

        vec = Matrices.flatten(point)
        point_ = self.space.reshape(vec)
        self.assertAllClose(point, point_, atol=atol)


class MatricesMetricTestCase(FlatRiemannianMetricTestCase):
    pass
