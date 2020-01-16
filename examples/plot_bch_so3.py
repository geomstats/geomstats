"""
Visualize the first, second and third order approximation of the Baker Campbell
Hausdorf formula on SO(3). To this end, sample 2 random elements of SO(3) and
"""
import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_orthogonal_group import SpecialOrthogonalGroup


n = 10
max_order = 10

group = SpecialOrthogonalGroup(n=n)
group.default_point_type = "matrix"

dim = int(n * (n - 1) / 2)
algebra = SkewSymmetricMatrices(n=n)


def main():
    norm_rv_1 = gs.normal(size=dim)
    tan_rv_1 = algebra.matrix_representation(
        norm_rv_1 / gs.norm(norm_rv_1, axis=0) / 2
    )
    exp_1 = gs.linalg.expm(tan_rv_1)

    norm_rv_2 = gs.normal(size=dim)
    tan_rv_2 = algebra.matrix_representation(
        norm_rv_2 / gs.norm(norm_rv_2, axis=0) / 2
    )
    exp_2 = gs.linalg.expm(tan_rv_2)

    composition = group.compose(exp_1, exp_2)

    orders = gs.arange(1, max_order)
    bch_approximations = gs.array(
        [
            algebra.baker_campbell_hausdorff(tan_rv_1, tan_rv_2, order=n)
            for n in orders
        ]
    )
    bch_approximations = algebra.basis_representation(bch_approximations)
    correct = algebra.basis_representation(gs.linalg.logm(composition))

    frobenius_error = gs.linalg.norm(bch_approximations - correct, axis=1)
    print(frobenius_error)

    fig, ax = plt.subplots()
    ax.scatter(orders, frobenius_error)

    ax.set(xlabel="Order of approximation", ylabel="Error in Frob. norm")
    ax.grid()

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
