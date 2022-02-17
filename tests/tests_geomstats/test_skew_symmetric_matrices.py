"""Unit tests for the skew symmetric matrices."""
import random

import geomstats.backend as gs
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from tests.conftest import TestCase
from tests.data_generation import MatrixLieAlgebraTestData
from tests.parametrizers import MatrixLieAlgebraParametrizer


class TestSkewSymmetricMatrices(TestCase, metaclass=MatrixLieAlgebraParametrizer):

    space = SkewSymmetricMatrices

    class TestDataSkewSymmetricMatrices(MatrixLieAlgebraTestData):
        n_list = [n for n in random.sample(range(2, 5), 2)]
        space_args_list = [(n,) for n in n_list]
        n_samples_list = [n for n in random.sample(range(2, 5), 2)]

        def belongs_data(self):
            smoke_data = [
                dict(n=2, mat=[[0.0, -1.0], [1.0, 0.0]], expected=True),
                dict(n=3, mat=[[0.0, -1.0], [1.0, 0.0]], expected=False),
            ]
            return self.generate_tests(smoke_data)

        def bch_up_to_fourth_order_works_data(self):
            smoke_data = [dict(n=2) for i in range(2, 10)]
            return self.generate_tests(smoke_data)

        def basis_representation_matrix_representation_composition_data(self):
            return self._basis_representation_matrix_representation_composition_data(
                SkewSymmetricMatrices, self.space_args_list, self.n_samples_list
            )

        def matrix_representation_basis_representation_composition(self):
            return self._matrix_representation_basis_representation_composition_data(
                SkewSymmetricMatrices, self.space_args_list, self.n_samples_list
            )

    testing_data = TestDataSkewSymmetricMatrices()

    def test_belongs(self, n, mat, expected):
        skew = self.space(n)
        self.assertAllClose(skew.belongs(gs.array(mat)), gs.array(expected))

    def test_bch_up_to_fourth_order_works(self, n):
        skew = SkewSymmetricMatrices(n)
        first_base = skew.basis[0]
        second_base = skew.basis[1]

        expected = first_base + second_base
        result = skew.baker_campbell_hausdorff(first_base, second_base, order=1)
        self.assertAllClose(expected, result)

        lb_first_second = skew.bracket(first_base, second_base)
        expected = expected + 0.5 * lb_first_second
        result = skew.baker_campbell_hausdorff(first_base, second_base, order=2)
        self.assertAllClose(expected, result)

        expected = (
            expected
            + 1.0 / 12.0 * skew.bracket(first_base, lb_first_second)
            - 1.0 / 12.0 * skew.bracket(second_base, lb_first_second)
        )
        result = skew.baker_campbell_hausdorff(first_base, second_base, order=3)
        self.assertAllClose(expected, result)

        expected = expected - 1.0 / 24.0 * skew.bracket(
            second_base, skew.bracket(first_base, lb_first_second)
        )
        result = skew.baker_campbell_hausdorff(first_base, second_base, order=4)
        self.assertAllClose(expected, result)
