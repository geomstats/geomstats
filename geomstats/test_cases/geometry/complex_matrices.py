from geomstats.geometry.complex_matrices import ComplexMatrixOperations
from geomstats.test_cases.geometry.base import MatrixVectorSpaceTestCase


class ComplexMatricesTestCase(MatrixVectorSpaceTestCase):
    def test_transconjugate(self, mat, expected, atol):
        res = ComplexMatrixOperations.transconjugate(mat)
        self.assertAllClose(res, expected, atol=atol)

    def test_is_hermitian(self, mat, expected, atol):
        res = ComplexMatrixOperations.is_hermitian(mat, atol=atol)
        self.assertAllEqual(res, expected)

    def test_is_hpd(self, mat, expected, atol):
        res = ComplexMatrixOperations.is_hpd(mat, atol=atol)
        self.assertAllEqual(res, expected)

    def test_to_hermitian(self, mat, expected, atol):
        res = ComplexMatrixOperations.to_hermitian(mat)
        self.assertAllClose(res, expected, atol=atol)
