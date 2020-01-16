"""
Unit tests for the skew symmetric matrices
"""
import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices


class TestSkewSymmetricMatrices(geomstats.tests.TestCase):
    def setUp(self):
        self.n_seq = gs.arange(3, 10)
        self.so = {n: SkewSymmetricMatrices(n=n) for n in self.n_seq}

    def test_basis_is_skew_symmetric(self):
        for n in self.n_seq:
            so = self.so[n]
            self.assertAllClose(
                so.basis + gs.transpose(so.basis, axes=(0, 2, 1)), 0
            )

    def test_basis_has_the_right_dimension(self):
        for n in self.n_seq:
            so = self.so[n]
            self.assertEqual(int(n * (n - 1) / 2), so.dimension)

    def test_bch_first_second_third_order_works(self):
        for n in self.n_seq:
            so = self.so[n]
            first_base = so.basis[0]
            second_base = so.basis[1]

            expected = first_base + second_base
            result = so.baker_campbell_hausdorff(
                first_base, second_base, order=1
            )
            self.assertAllClose(expected, result)

            lb_first_second = so.lie_bracket(first_base, second_base)
            expected = expected + 0.5 * lb_first_second
            result = so.baker_campbell_hausdorff(
                first_base, second_base, order=2
            )
            self.assertAllClose(expected, result)

            expected = (
                expected
                + 1.0 / 12.0 * so.lie_bracket(first_base, lb_first_second)
                - 1.0 / 12.0 * so.lie_bracket(second_base, lb_first_second)
            )
            result = so.baker_campbell_hausdorff(
                first_base, second_base, order=3
            )
            self.assertAllClose(expected, result)

            expected = expected - 1.0 / 24.0 * so.lie_bracket(
                second_base, so.lie_bracket(first_base, lb_first_second)
            )
            result = so.baker_campbell_hausdorff(
                first_base, second_base, order=4
            )
            self.assertAllClose(expected, result)

    def test_basis_representation_is_correctly_vectorized(self):
        for n in self.n_seq:
            so = self.so[n]
            shape = gs.shape(so.basis_representation(so.basis))
            dim = int(n * (n - 1) / 2)
            self.assertEqual(shape, (dim, dim))
