from geomstats.test_cases.geometry.base import ComplexVectorSpaceTestCase


class ComplexMatricesTestCase(ComplexVectorSpaceTestCase):
    # TODO: with mixins?

    def test_transconjugate(self, mat, expected, atol):
        res = self.space.transconjugate(mat)
        self.assertAllClose(res, expected, atol=atol)

    def test_is_hermitian(self, mat, expected, atol):
        res = self.space.is_hermitian(mat, atol=atol)
        self.assertAllEqual(res, expected)

    def test_is_hpd(self, mat, expected, atol):
        res = self.space.is_hpd(mat, atol=atol)
        self.assertAllEqual(res, expected)

    def test_to_hermitian(self, mat, expected, atol):
        res = self.space.to_hermitian(mat)
        self.assertAllClose(res, expected, atol=atol)
