import geomstats.backend as gs
from geomstats.test_cases.geometry.lie_algebra import MatrixLieAlgebraTestCase


class SkewSymmetricMatricesTestCase(MatrixLieAlgebraTestCase):
    def test_baker_campbell_hausdorff_with_basis(self, atol=gs.atol):
        """Check Baker-Campbell-Hausdorff approximation on basis elements.

        Compares the approximations of orders 1 to 4 for the first two basis
        elements against their expansion in terms of brackets.

        Parameters
        ----------
        atol : float
            Absolute tolerance.

        Notes
        -----
        Only valid for n > 2.
        """
        if self.space.n < 3:
            return

        fb = self.space.basis[0]
        sb = self.space.basis[1]

        fb_sb_bracket = self.space.bracket(fb, sb)
        expected1 = fb + sb
        expected2 = expected1 + 0.5 * fb_sb_bracket
        expected3 = (
            expected2
            + 1.0 / 12.0 * self.space.bracket(fb, fb_sb_bracket)
            - 1.0 / 12.0 * self.space.bracket(sb, fb_sb_bracket)
        )
        expected4 = expected3 - 1.0 / 24.0 * self.space.bracket(
            sb, self.space.bracket(fb, fb_sb_bracket)
        )
        expected = [expected1, expected2, expected3, expected4]
        for order in range(1, 5):
            self.test_baker_campbell_hausdorff(
                matrix_a=fb,
                matrix_b=sb,
                order=order,
                expected=expected[order - 1],
                atol=atol,
            )
