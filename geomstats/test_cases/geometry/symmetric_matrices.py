from geomstats.test.geometry.base import (
    MatrixVectorSpaceTestCaseMixins,
    VectorSpaceTestCase,
)
from geomstats.test.test_case import TestCase

# TODO: mixins with MatrixVectorSpaces?
# TODO: use `self.space.ndim` to control vector dimension
# TODO: check if from vector gives same order as matrix


class SymmetricMatricesTestCase(MatrixVectorSpaceTestCaseMixins, VectorSpaceTestCase):
    pass


class SymmetricMatricesOpsTestCase(TestCase):
    # TODO: check apply_func_to_eigenvals alone

    def test_expm(self, mat, expected, atol):
        res = self.Space.expm(mat)
        self.assertAllClose(res, expected, atol=atol)

    def test_powerm(self, mat, power, expected, atol):
        # TODO: check vectorization
        res = self.Space.powerm(mat, power)
        self.assertAllClose(res, expected, atol=atol)
