import random

from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from tests.data_generation import _MatrixLieAlgebraTestData


class SkewSymmetricMatricesTestData(_MatrixLieAlgebraTestData):
    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    shape_list = [(n, n) for n in n_list]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    space = SkewSymmetricMatrices

    def belongs_test_data(self):
        smoke_data = [
            dict(n=2, mat=[[0.0, -1.0], [1.0, 0.0]], expected=True),
            dict(n=3, mat=[[0.0, -1.0], [1.0, 0.0]], expected=False),
        ]
        return self.generate_tests(smoke_data)

    def baker_campbell_hausdorff_test_data(self):
        n_list = range(3, 10)
        smoke_data = []
        for n in n_list:
            space = SkewSymmetricMatrices(n)
            fb = space.basis[0]
            sb = space.basis[1]
            fb_sb_bracket = space.bracket(fb, sb)
            expected1 = fb + sb
            expected2 = expected1 + 0.5 * fb_sb_bracket
            expected3 = (
                expected2
                + 1.0 / 12.0 * space.bracket(fb, fb_sb_bracket)
                - 1.0 / 12.0 * space.bracket(sb, fb_sb_bracket)
            )
            expected4 = expected3 - 1.0 / 24.0 * space.bracket(
                sb, space.bracket(fb, fb_sb_bracket)
            )
            expected = [expected1, expected2, expected3, expected4]
            for order in range(1, 5):
                smoke_data.append(
                    dict(
                        n=n,
                        matrix_a=fb,
                        matrix_b=sb,
                        order=order,
                        expected=expected[order - 1],
                    )
                )

        return self.generate_tests(smoke_data)
