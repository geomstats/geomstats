"""Visualize convergence of the BCH formula approximation on so(n).

Visualize the first, second and third order approximation of the Baker Campbell
Hausdorff formula on so(n). To this end, sample 2 random elements a,b of so(n)
and compute both the BCH approximations of different orders as well as
log(exp(a)exp(b)) and compare these in the Frobenius norm.

Notice that the BCH only guarantees convergence if ||a|| + ||b|| < log 2,
so we normalize the random vectors to have norm 1 / 2.

We also compare execution times of the scikit-learn expm / logm implementation
with our BCH-implementation, for small orders approximation by BCH is faster
than the scikit-learn version, while being close to the actual value.

"""
import timeit

import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

N = 3
MAX_ORDER = 10

GROUP = SpecialOrthogonal(n=N)

DIM = int(N * (N - 1) / 2)
ALGEBRA = SkewSymmetricMatrices(n=N)


def main():
    """Visualize convergence of the BCH formula approximation on so(n)."""
    norm_rv_1 = gs.random.normal(size=DIM)
    tan_rv_1 = ALGEBRA.matrix_representation(
        norm_rv_1 / gs.linalg.norm(norm_rv_1, axis=0) / 2
    )
    exp_1 = gs.linalg.expm(tan_rv_1)

    norm_rv_2 = gs.random.normal(size=DIM)
    tan_rv_2 = ALGEBRA.matrix_representation(
        norm_rv_2 / gs.linalg.norm(norm_rv_2, axis=0) / 2
    )
    exp_2 = gs.linalg.expm(tan_rv_2)

    composition = GROUP.compose(exp_1, exp_2)

    orders = gs.arange(1, MAX_ORDER + 1)
    bch_approximations = gs.array(
        [ALGEBRA.baker_campbell_hausdorff(tan_rv_1, tan_rv_2, order=n) for n in orders]
    )
    bch_approximations = ALGEBRA.basis_representation(bch_approximations)
    correct = ALGEBRA.basis_representation(gs.linalg.logm(composition))
    t_numpy = timeit.timeit(
        lambda: gs.linalg.logm(
            gs.matmul(gs.linalg.expm(tan_rv_1), gs.linalg.expm(tan_rv_2))
        ),
        number=100,
    )
    t_bch = [
        timeit.timeit(
            lambda order=n: ALGEBRA.baker_campbell_hausdorff(
                tan_rv_1, tan_rv_2, order=order
            ),
            number=100,
        )
        for n in orders
    ]
    frobenius_error = gs.linalg.norm(bch_approximations - correct, axis=1)

    plt.subplot(2, 1, 1)
    plt.scatter(orders, frobenius_error)
    plt.xlabel("Order of approximation")
    plt.ylabel("Error in Frob. norm")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.scatter(orders, t_bch)
    plt.hlines(y=t_numpy, xmin=1, xmax=MAX_ORDER)
    plt.xlabel("Order of approximation")
    plt.ylabel("Execution time[s] for 100 replications vs. numpy")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
