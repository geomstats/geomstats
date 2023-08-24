import pytest

import geomstats.backend as gs
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import VectorSpaceTestCase


class MatrixLieAlgebraTestCase(VectorSpaceTestCase):
    # most of tests are very similar to MatrixVectorSpaceTestCaseMixins

    # TODO: any mathematical property for baker_campbell_hausdorff?

    def test_baker_campbell_hausdorff(
        self, matrix_a, matrix_b, expected, atol, order=2
    ):
        res = self.space.baker_campbell_hausdorff(matrix_a, matrix_b, order=order)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_baker_campbell_hausdorff_vec(self, n_reps, atol, order=2):
        matrix_a, matrix_b = self.data_generator.random_point(2)
        expected = self.space.baker_campbell_hausdorff(matrix_a, matrix_b, order=order)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    matrix_a=matrix_a,
                    matrix_b=matrix_b,
                    expected=expected,
                    atol=atol,
                    order=order,
                )
            ],
            arg_names=["matrix_a", "matrix_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_basis_representation(self, matrix_representation, expected, atol):
        vec = self.space.basis_representation(matrix_representation)
        self.assertAllClose(vec, expected, atol=atol)

    @pytest.mark.vec
    def test_basis_representation_vec(self, n_reps, atol):
        matrix_representation = self.data_generator.random_point()
        expected = self.space.basis_representation(matrix_representation)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    matrix_representation=matrix_representation,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["matrix_representation"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_basis_representation_and_basis(self, n_points, atol):
        mat = self.data_generator.random_point(n_points)
        vec = self.space.basis_representation(mat)

        res = gs.einsum("...i,...ijk->...jk", vec, self.space.basis)
        self.assertAllClose(res, mat, atol=atol)

    def test_matrix_representation(self, vec, expected, atol):
        mat = self.space.matrix_representation(vec)
        self.assertAllClose(mat, expected, atol=atol)

    @pytest.mark.vec
    def test_matrix_representation_vec(self, n_reps, atol):
        vec = gs.random.rand(self.space.dim)
        expected = self.space.matrix_representation(vec)

        vec_data = generate_vectorization_data(
            data=[dict(vec=vec, expected=expected, atol=atol)],
            arg_names=["vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_matrix_representation_belongs(self, n_points, atol):
        vec = gs.reshape(gs.random.rand(n_points * self.space.dim), (n_points, -1))
        point = self.space.matrix_representation(vec)

        self.test_belongs(point, gs.ones(n_points, dtype=bool), atol)

    @pytest.mark.random
    def test_matrix_representation_after_basis_representation(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        vec = self.space.basis_representation(point)
        point_ = self.space.matrix_representation(vec)

        self.assertAllClose(point_, point, atol=atol)

    @pytest.mark.random
    def test_basis_representation_after_matrix_representation(self, n_points, atol):
        vec = gs.reshape(gs.random.rand(n_points * self.space.dim), (n_points, -1))
        point = self.space.matrix_representation(vec)
        vec_ = self.space.basis_representation(point)

        self.assertAllClose(vec_, vec, atol=atol)
