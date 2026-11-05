import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.base import MatrixVectorSpaceTestCase
from geomstats.test_cases.geometry.euclidean import EuclideanMetricTestCase


class MatricesTestCase(MatrixVectorSpaceTestCase):
    def test_reshape_after_flatten(self, point, atol=gs.atol):
        """Check reshape is a left inverse of flatten.

        Parameters
        ----------
        point : array-like, shape=[..., m, n]
            Matrix.
        atol : float
            Absolute tolerance.
        """
        vec = Matrices.flatten(point)
        point_ = self.space.reshape(vec)
        self.assertAllClose(point_, point, atol=atol)


class MatricesMetricTestCase(EuclideanMetricTestCase):
    pass


class MatrixOperationsTestCase(TestCase):
    def test_equal(self, mat_a, mat_b, expected, atol=gs.atol):
        """Check matrix equality.

        Parameters
        ----------
        mat_a : array-like, shape=[..., m, n]
            Matrix.
        mat_b : array-like, shape=[..., m, n]
            Matrix.
        expected : array-like, shape=[...,]
            Boolean evaluating if the matrices are equal.
        atol : float
            Absolute tolerance.
        """
        res = Matrices.equal(mat_a, mat_b, atol=atol)
        self.assertAllEqual(res, expected)

    def test_mul_reduce(self, mats, expected, atol=gs.atol):
        """Check product of several matrices.

        Parameters
        ----------
        mats : tuple[array-like]
            Matrices with compatible shapes.
        expected : array-like, shape=[..., m, k]
            Expected product.
        atol : float
            Absolute tolerance.
        """
        res = Matrices.mul(*mats)
        self.assertAllClose(res, expected, atol=atol)

    def test_mul(self, mat_a, mat_b, expected, atol=gs.atol):
        """Check matrix product.

        Parameters
        ----------
        mat_a : array-like, shape=[..., m, n]
            Matrix.
        mat_b : array-like, shape=[..., n, k]
            Matrix.
        expected : array-like, shape=[..., m, k]
            Expected product.
        atol : float
            Absolute tolerance.
        """
        res = Matrices.mul(mat_a, mat_b)
        self.assertAllClose(res, expected, atol=atol)

    def test_bracket(self, mat_a, mat_b, expected, atol=gs.atol):
        """Check commutator.

        Parameters
        ----------
        mat_a : array-like, shape=[..., n, n]
            Square matrix.
        mat_b : array-like, shape=[..., n, n]
            Square matrix.
        expected : array-like, shape=[..., n, n]
            Expected commutator.
        atol : float
            Absolute tolerance.
        """
        res = Matrices.bracket(mat_a, mat_b)
        self.assertAllClose(res, expected, atol=atol)

    def test_transpose(self, mat, expected, atol=gs.atol):
        """Check transpose.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Matrix.
        expected : array-like, shape=[..., n, m]
            Expected transpose.
        atol : float
            Absolute tolerance.
        """
        res = gs.transpose(mat)
        self.assertAllClose(res, expected, atol=atol)

    def test_diagonal(self, mat, expected, atol=gs.atol):
        """Check diagonal.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Matrix.
        expected : array-like, shape=[..., min(m, n)]
            Expected diagonal.
        atol : float
            Absolute tolerance.
        """
        res = Matrices.diagonal(mat)
        self.assertAllClose(res, expected, atol=atol)

    def test_is_square(self, mat, expected):
        """Check if matrix is square.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Matrix.
        expected : array-like, shape=[...,]
            Boolean evaluating if `mat` is square.
        """
        res = Matrices.is_square(mat)
        self.assertAllEqual(res, expected)

    def test_is_property(self, property_name, mat, expected, atol=gs.atol):
        """Check if matrix has a property.

        Parameters
        ----------
        property_name : str
            Name of the property, as in ``Matrices.is_{property_name}``.
        mat : array-like, shape=[..., m, n]
            Matrix.
        expected : array-like, shape=[...,]
            Boolean evaluating if `mat` has the property.
        atol : float
            Absolute tolerance.
        """
        func = getattr(Matrices, f"is_{property_name}")
        res = func(mat, atol=atol)
        self.assertAllEqual(res, expected)

    def test_congruent(self, mat_1, mat_2, expected, atol=gs.atol):
        """Check congruence.

        Parameters
        ----------
        mat_1 : array-like, shape=[..., n, n]
            Square matrix.
        mat_2 : array-like, shape=[..., n, n]
            Square matrix.
        expected : array-like, shape=[..., n, n]
            Expected congruent matrix.
        atol : float
            Absolute tolerance.
        """
        res = Matrices.congruent(mat_1, mat_2)
        self.assertAllClose(res, expected, atol=atol)

    def test_frobenius_product(self, mat_1, mat_2, expected, atol=gs.atol):
        """Check Frobenius product.

        Parameters
        ----------
        mat_1 : array-like, shape=[..., m, n]
            Matrix.
        mat_2 : array-like, shape=[..., m, n]
            Matrix.
        expected : array-like, shape=[...,]
            Expected Frobenius product.
        atol : float
            Absolute tolerance.
        """
        res = Matrices.frobenius_product(mat_1, mat_2)
        self.assertAllClose(res, expected, atol=atol)

    def test_trace_product(self, mat_1, mat_2, expected, atol=gs.atol):
        """Check trace of the product.

        Parameters
        ----------
        mat_1 : array-like, shape=[..., m, n]
            Matrix.
        mat_2 : array-like, shape=[..., n, m]
            Matrix.
        expected : array-like, shape=[...,]
            Expected trace of the product.
        atol : float
            Absolute tolerance.
        """
        res = Matrices.trace_product(mat_1, mat_2)
        self.assertAllClose(res, expected, atol=atol)

    def test_flatten(self, mat, expected, atol=gs.atol):
        """Check flattening.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Matrix.
        expected : array-like, shape=[..., m * n]
            Expected flattened matrix.
        atol : float
            Absolute tolerance.
        """
        res = Matrices.flatten(mat)
        self.assertAllClose(res, expected, atol=atol)

    def test_align_matrices(self, point, base_point, expected, atol=gs.atol):
        """Check matrix alignment.

        Parameters
        ----------
        point : array-like, shape=[..., m, n]
            Matrix to align.
        base_point : array-like, shape=[..., m, n]
            Matrix to align to.
        expected : array-like, shape=[..., m, n]
            Expected aligned matrix.
        atol : float
            Absolute tolerance.
        """
        res = Matrices.align_matrices(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_equal_true(self, mat, atol=gs.atol):
        """Check a matrix is equal to itself.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Matrix.
        atol : float
            Absolute tolerance.
        """
        self.test_equal(mat, mat, True, atol=atol)

    def test_equal_false(self, mat_a, mat_b, atol=gs.atol):
        """Check different random matrices are not equal.

        Parameters
        ----------
        mat_a : array-like, shape=[..., m, n]
            Matrix.
        mat_b : array-like, shape=[..., m, n]
            Matrix.
        atol : float
            Absolute tolerance.
        """
        self.test_equal(mat_a, mat_b, False, atol=atol)

    def test_mul_identity(self, mat, atol=gs.atol):
        """Check identity is a right identity of the product.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Matrix.
        atol : float
            Absolute tolerance.
        """
        identity = gs.eye(mat.shape[-1])

        self.test_mul(mat, identity, mat, atol)

    def test_bracket_bilinearity(self, mat_a, mat_b, mat_c, coefficients, atol=gs.atol):
        """Check commutator is bilinear.

        Parameters
        ----------
        mat_a : array-like, shape=[..., n, n]
            Square matrix.
        mat_b : array-like, shape=[..., n, n]
            Square matrix.
        mat_c : array-like, shape=[..., n, n]
            Square matrix.
        coefficients : array-like, shape=[2]
            Coefficients of the linear combination.
        atol : float
            Absolute tolerance.
        """
        a, b = coefficients

        mat_ab = a * mat_a + b * mat_b
        left = Matrices.bracket(mat_ab, mat_c)
        right = a * Matrices.bracket(mat_a, mat_c) + b * Matrices.bracket(mat_b, mat_c)
        self.assertAllClose(left, right, atol=atol)

        left = Matrices.bracket(mat_c, mat_ab)
        right = a * Matrices.bracket(mat_c, mat_a) + b * Matrices.bracket(mat_c, mat_b)
        self.assertAllClose(left, right, atol=atol)

    def test_transpose_sym(self, mat, atol=gs.atol):
        """Check a symmetric matrix is its own transpose.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Symmetric matrix.
        atol : float
            Absolute tolerance.
        """
        self.test_transpose(mat, mat, atol)

    def test_diagonal_sum(self, mat, atol=gs.atol):
        """Check sum of the diagonal is the trace.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Square matrix.
        atol : float
            Absolute tolerance.
        """
        diag = Matrices.diagonal(mat)

        sum_diag = gs.sum(diag, axis=-1)
        trace = gs.trace(mat)
        self.assertAllClose(sum_diag, trace, atol=atol)

    def test_is_square_true(self, mat):
        """Check a square matrix is square.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Square matrix.
        """
        self.test_is_square(mat, True)

    def test_is_square_false(self, mat):
        """Check a non-square matrix is not square.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Non-square matrix.
        """
        self.test_is_square(mat, False)

    def test_is_property_true(self, property_name, mat, atol=gs.atol):
        """Check a matrix sampled with a property has it.

        Parameters
        ----------
        property_name : str
            Name of the property, as in ``Matrices.is_{property_name}``.
        mat : array-like, shape=[..., m, n]
            Matrix.
        atol : float
            Absolute tolerance.
        """
        self.test_is_property(property_name, mat, True, atol)

    def test_is_property_false_square(self, property_name, mat, atol=gs.atol):
        """Check a random square matrix does not have a property.

        Parameters
        ----------
        property_name : str
            Name of the property, as in ``Matrices.is_{property_name}``.
        mat : array-like, shape=[..., n, n]
            Square matrix.
        atol : float
            Absolute tolerance.
        """
        self.test_is_property(property_name, mat, False, atol)

    def test_to_property_is_property(self, property_name, mat, atol=gs.atol):
        """Check projecting to a property gives a matrix with it.

        Parameters
        ----------
        property_name : str
            Name of the property, as in ``Matrices.to_{property_name}``.
        mat : array-like, shape=[..., n, n]
            Square matrix.
        atol : float
            Absolute tolerance.
        """
        to_func = getattr(Matrices, f"to_{property_name}")
        is_func = getattr(Matrices, f"is_{property_name}")

        res = is_func(to_func(mat), atol=atol)
        self.assertAllEqual(res, True)

    def test_to_lower_triangular_diagonal_scaled_is_lower_triangular(
        self, mat, atol=gs.atol
    ):
        """Check diagonally scaled lower triangular projection is lower triangular.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Square matrix.
        atol : float
            Absolute tolerance.
        """
        res = Matrices.is_lower_triangular(
            Matrices.to_lower_triangular_diagonal_scaled(mat), atol=atol
        )
        self.assertAllEqual(res, True)
