"""
Visualize the first, second and third order approximation of the Baker Campbell
Hausdorf formula on SO(3). To this end, sample 2 random elements of SO(3) and
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np

import geomstats.visualization as visualization
import geomstats.backend as gs

from geomstats.learning.pca import TangentPCA
from geomstats.geometry.special_orthogonal_group import SpecialOrthogonalGroup
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices

dim = 3
group = SpecialOrthogonalGroup(n=dim)
group.default_point_type = 'matrix'
algebra = SkewSymmetricMatrices(dimension=dim, n=dim)

def main():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    tangent_vecs = algebra.matrix_representation(
            2 * (gs.random.rand(2,dim) - 0.5))
    exponentials = group.group_exp(tangent_vecs)

    composition = group.compose(exponentials[0], exponentials[1])

    bch_approximations = gs.array([
        algebra.baker_campbell_hausdorff(
            tangent_vecs[0], tangent_vecs[1], order=n
            )
        for n in gs.arange(1,20)
        ])
    bch_approximations = algebra.basis_representation(bch_approximations)
    correct = algebra.basis_representation(group.group_log(composition))

    ax.scatter(bch_approximations[:, 0],
            bch_approximations[:, 1],
            bch_approximations[:, 2])
    ax.scatter(correct[:, 0], correct[:, 1], correct[:, 2],
            color = "red", s=200)

    plt.show()

if __name__ == "__main__":
    main()
