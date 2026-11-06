import geomstats.backend as gs
from geomstats.test_cases.geometry.base import MatrixVectorSpaceTestCase


class MatrixLieAlgebraTestCase(MatrixVectorSpaceTestCase):
    def test_baker_campbell_hausdorff(
        self, matrix_a, matrix_b, expected, atol=gs.atol, order=2
    ):
        """Check Baker-Campbell-Hausdorff approximation.

        Parameters
        ----------
        matrix_a : array-like, shape=[..., n, n]
            Matrix in the Lie algebra.
        matrix_b : array-like, shape=[..., n, n]
            Matrix in the Lie algebra.
        expected : array-like, shape=[..., n, n]
            Expected approximation of log(exp(matrix_a) exp(matrix_b)).
        atol : float
            Absolute tolerance.
        order : int
            Truncation order of the approximation.
        """
        res = self.space.baker_campbell_hausdorff(matrix_a, matrix_b, order=order)
        self.assertAllClose(res, expected, atol=atol)
