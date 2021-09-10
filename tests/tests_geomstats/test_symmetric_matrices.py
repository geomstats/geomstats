"""Unit tests for the vector space of symmetric matrices."""

import math
import warnings

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


class TestSymmetricMatrices(geomstats.tests.TestCase):
    """Test of SymmetricMatrices methods."""

    def setUp(self):
        """Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.n = 3
        self.space = SymmetricMatrices(self.n)

    def test_belongs(self):
        """Test of belongs method."""
        sym_n = self.space
        mat_sym = gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        mat_not_sym = gs.array([[1.0, 0.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        result = sym_n.belongs(mat_sym)
        expected = True
        self.assertAllClose(result, expected)

        result = sym_n.belongs(mat_not_sym)
        expected = False
        self.assertAllClose(result, expected)

    def test_basis(self):
        """Test of belongs method."""
        sym_n = SymmetricMatrices(2)
        mat_sym_1 = gs.array([[1.0, 0.0], [0, 0]])
        mat_sym_2 = gs.array([[0, 1.0], [1.0, 0]])
        mat_sym_3 = gs.array([[0, 0.0], [0, 1.0]])
        expected = gs.stack([mat_sym_1, mat_sym_2, mat_sym_3])
        result = sym_n.basis
        self.assertAllClose(result, expected)

    def test_expm(self):
        """Test of expm method."""
        sym_n = SymmetricMatrices(self.n)
        v = gs.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = sym_n.expm(v)
        c = math.cosh(1)
        s = math.sinh(1)
        e = math.exp(1)
        expected = gs.array([[c, s, 0.0], [s, c, 0.0], [0.0, 0.0, e]])

        four_dim_v = gs.broadcast_to(v, (2, 2) + v.shape)
        four_dim_expected = gs.broadcast_to(expected, (2, 2) + expected.shape)
        four_dim_result = sym_n.expm(four_dim_v)

        self.assertAllClose(result, expected)
        self.assertAllClose(four_dim_result, four_dim_expected)

    def test_powerm(self):
        """Test of powerm method."""
        sym_n = SymmetricMatrices(self.n)
        expected = gs.array(
            [[[1, 1.0 / 4.0, 0.0], [1.0 / 4, 2.0, 0.0], [0.0, 0.0, 1.0]]]
        )

        power = gs.array(1.0 / 2.0)

        result = sym_n.powerm(expected, power)
        result = gs.matmul(result, gs.transpose(result, (0, 2, 1)))
        self.assertAllClose(result, expected)

    def test_vector_from_symmetric_matrix_and_symmetric_matrix_from_vector(self):
        """Test for matrix to vector and vector to matrix conversions."""
        sym_mat_1 = gs.array([[1.0, 0.6, -3.0], [0.6, 7.0, 0.0], [-3.0, 0.0, 8.0]])
        vector_1 = self.space.to_vector(sym_mat_1)
        result_1 = self.space.from_vector(vector_1)
        expected_1 = sym_mat_1

        self.assertTrue(gs.allclose(result_1, expected_1))

        vector_2 = gs.array([1, 2, 3, 4, 5, 6])
        sym_mat_2 = self.space.from_vector(vector_2)
        result_2 = self.space.to_vector(sym_mat_2)
        expected_2 = vector_2

        self.assertTrue(gs.allclose(result_2, expected_2))

    def test_vector_and_symmetric_matrix_vectorization(self):
        """Test of vectorization."""
        n_samples = 5
        vector = gs.random.rand(n_samples, 6)
        sym_mat = self.space.from_vector(vector)
        result = self.space.to_vector(sym_mat)
        expected = vector

        self.assertTrue(gs.allclose(result, expected))

        vector = self.space.to_vector(sym_mat)
        result = self.space.from_vector(vector)
        expected = sym_mat

        self.assertTrue(gs.allclose(result, expected))

    def test_symmetric_matrix_from_vector(self):
        vector_2 = gs.array([1, 2, 3, 4, 5, 6])
        result = self.space.from_vector(vector_2)
        expected = gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        self.assertAllClose(result, expected)

    def test_projection_and_belongs(self):
        shape = (2, self.n, self.n)
        result = helper.test_projection_and_belongs(self.space, shape)
        for res in result:
            self.assertTrue(res)

    def test_random_and_belongs(self):
        mat = self.space.random_point()
        result = self.space.belongs(mat)
        self.assertTrue(result)

    def test_dim(self):
        result = self.space.dim
        n = self.space.n
        expected = int(n * (n + 1) / 2)
        self.assertAllClose(result, expected)
