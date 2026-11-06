import geomstats.backend as gs
from geomstats.geometry.complex_matrices import ComplexMatrices
from geomstats.test.test_case import TestCase


class ComplexMatrixOperationsTestCase(TestCase):
    def test_transconjugate(self, mat, expected, atol=gs.atol):
        """Check transconjugate.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Matrix.
        expected : array-like, shape=[..., n, m]
            Expected conjugate transpose.
        atol : float
            Absolute tolerance.
        """
        res = ComplexMatrices.transconjugate(mat)
        self.assertAllClose(res, expected, atol=atol)

    def test_is_property(self, property_name, mat, expected, atol=gs.atol):
        """Check if matrix has a property.

        Parameters
        ----------
        property_name : str
            Name of the property, as in ``ComplexMatrices.is_{property_name}``.
        mat : array-like, shape=[..., n, n]
            Matrix.
        expected : array-like, shape=[...,]
            Boolean evaluating if `mat` has the property.
        atol : float
            Absolute tolerance.
        """
        func = getattr(ComplexMatrices, f"is_{property_name}")
        res = func(mat, atol=atol)
        self.assertAllEqual(res, expected)

    def test_to_hermitian(self, mat, expected, atol=gs.atol):
        """Check Hermitian part of a matrix.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        expected : array-like, shape=[..., n, n]
            Expected Hermitian matrix.
        atol : float
            Absolute tolerance.
        """
        res = ComplexMatrices.to_hermitian(mat)
        self.assertAllClose(res, expected, atol=atol)

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
        res = ComplexMatrices.congruent(mat_1, mat_2)
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
        res = ComplexMatrices.frobenius_product(mat_1, mat_2)
        self.assertAllClose(res, expected, atol=atol)

    def test_is_property_true(self, property_name, mat, atol=gs.atol):
        """Check a matrix sampled with a property has it.

        Parameters
        ----------
        property_name : str
            Name of the property, as in ``ComplexMatrices.is_{property_name}``.
        mat : array-like, shape=[..., n, n]
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
            Name of the property, as in ``ComplexMatrices.is_{property_name}``.
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
            Name of the property, as in ``ComplexMatrices.to_{property_name}``.
        mat : array-like, shape=[..., n, n]
            Square matrix.
        atol : float
            Absolute tolerance.
        """
        to_func = getattr(ComplexMatrices, f"to_{property_name}")
        is_func = getattr(ComplexMatrices, f"is_{property_name}")

        res = is_func(to_func(mat), atol=atol)
        self.assertAllEqual(res, True)
