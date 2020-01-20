"""
Unit tests for the skew symmetric matrices
"""
import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices


class TestSkewSymmetricMatrices(geomstats.tests.TestCase):
    @geomstats.tests.np_and_pytorch_only
    def setUp(self):
        self.n_seq = [3, 4, 5, 6, 7, 8, 9, 10]
        self.skew = {n: SkewSymmetricMatrices(n=n) for n in self.n_seq}

    @geomstats.tests.np_and_pytorch_only
    def test_basis_is_skew_symmetric(self):
        for n in self.n_seq:
            skew = self.skew[n]
            self.assertAllClose(
                skew.basis + gs.transpose(skew.basis, axes=(0, 2, 1)), 0
            )

    @geomstats.tests.np_and_pytorch_only
    def test_basis_has_the_right_dimension(self):
        for n in self.n_seq:
            skew = self.skew[n]
            self.assertEqual(int(n * (n - 1) / 2), skew.dimension)

    @geomstats.tests.np_and_pytorch_only
    def test_bch_up_to_fourth_order_works(self):
        for n in self.n_seq:
            skew = self.skew[n]
            first_base = skew.basis[0]
            second_base = skew.basis[1]

            expected = first_base + second_base
            result = skew.baker_campbell_hausdorff(
                first_base, second_base, order=1
            )
            self.assertAllClose(expected, result)

            lb_first_second = skew.lie_bracket(first_base, second_base)
            expected = expected + 0.5 * lb_first_second
            result = skew.baker_campbell_hausdorff(
                first_base, second_base, order=2
            )
            self.assertAllClose(expected, result)

            expected = (
                expected
                + 1.0 / 12.0 * skew.lie_bracket(first_base, lb_first_second)
                - 1.0 / 12.0 * skew.lie_bracket(second_base, lb_first_second)
            )
            result = skew.baker_campbell_hausdorff(
                first_base, second_base, order=3
            )
            self.assertAllClose(expected, result)

            expected = expected - 1.0 / 24.0 * skew.lie_bracket(
                second_base, skew.lie_bracket(first_base, lb_first_second)
            )
            result = skew.baker_campbell_hausdorff(
                first_base, second_base, order=4
            )
            self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_basis_representation_is_correctly_vectorized(self):
        for n in self.n_seq:
            skew = self.skew[n]
            shape = gs.shape(skew.basis_representation(skew.basis))
            dim = int(n * (n - 1) / 2)
            self.assertEqual(shape, (dim, dim))
