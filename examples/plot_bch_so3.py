"""
Visualize the first, second and third order approximation of the Baker Campbell
Hausdorf formula on SO(3). To this end, sample 2 random elements of SO(3) and
"""

import geomstats.backend as gs
from geomstats.geometry.matrices_space import MatricesMetric
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_orthogonal_group import SpecialOrthogonalGroup

n = 3
group = SpecialOrthogonalGroup(n=n)
group.default_point_type = 'matrix'
dim = int(n * (n - 1) / 2)
algebra = SkewSymmetricMatrices(dimension=dim, n=n)


def main():
    tangent_vecs = algebra.matrix_representation(
            2 * (gs.random.rand(2, dim) - 0.5))
    exponentials = group.group_exp(tangent_vecs)

    composition = group.compose(exponentials[0], exponentials[1])

    bch_approximations = gs.array([
        algebra.baker_campbell_hausdorff(
            tangent_vecs[0], tangent_vecs[1], order=n
            )
        for n in gs.arange(1, 20)
        ])
    bch_approximations = algebra.basis_representation(bch_approximations)
    correct = algebra.basis_representation(group.group_log(composition))

    metric = MatricesMetric(n, n)

    print(metric.dist(correct, bch_approximations))


if __name__ == "__main__":
    main()
