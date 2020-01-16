import geomstats.backend as gs

bch_info = gs.array(
    [
        [int(x) for x in i.strip().split()]
        for i in open("geomstats/geometry/bchHall20.dat").readlines()
    ]
)


class MatrixLieAlgebra:
    """
    Class implementing matrix Lie algebra related functions.
    """

    def __init__(self, dimension, n):
        self.dimension = dimension
        self.n = n
        self.basis = None

    def lie_bracket(self, matrix_a, matrix_b):
        """
        Parameters
        ----------
        matrix_a: array-like, shape=[n_sample, n, n]
        matrix_b: array-like, shape=[n_sample, n, n]

        Returns
        -------
        bracket: shape=[n_sample, n, n]

        """
        return gs.matmul(matrix_a, matrix_b) - gs.matmul(matrix_b, matrix_a)

    def baker_campbell_hausdorff(self, matrix_a, matrix_b, order=2):
        """
        Calculates the Baker Campbell Hausdorff approximation up to
        the given order.

        We use the algorithm published by Casas / Murua in their paper
        "An efficient algorithm for computing the Baker–Campbell–Hausdorff
        series and some of its applications" in J.o.Math.Physics 50 (2009).

        This represents Z =log(exp(X)exp(Y)) as an infinite linear combination
        of the form
            Z = sum z_i E_i
        where z_i are rational numbers and E_i are iterated Lie brackets
        starting with E_1 = X, E_2 = Y, each E_i is given by some i',i'':
            E_i = [E_i', E_i''].

        Parameters
        ----------
        matrix_a: array-like, shape=[n_sample, n, n]
        matrix_b: array-like, shape=[n_sample, n, n]
        order: int
            the order to which the approximation is calculated. Note that this
            is NOT the same as using only E_i with i < order
        """

        number_of_hom_degree = gs.array(
            [
                2,
                1,
                2,
                3,
                6,
                9,
                18,
                30,
                56,
                99,
                186,
                335,
                630,
                1161,
                2182,
                4080,
                7710,
                14532,
                27594,
                52377,
            ]
        )
        n_terms = gs.sum(number_of_hom_degree[:order])

        Ei = gs.zeros((n_terms, self.n, self.n))
        Ei[0] = matrix_a
        Ei[1] = matrix_b
        result = matrix_a + matrix_b

        for i in gs.arange(2, n_terms):
            i_p = bch_info[i, 1] - 1
            i_pp = bch_info[i, 2] - 1

            Ei[i] = self.lie_bracket(Ei[i_p], Ei[i_pp])
            result = result + bch_info[i, 3] / float(bch_info[i, 4]) * Ei[i]

        return result

    def basis_representation(self, matrix_representation):
        """
        Parameters
        ----------
        matrix_representation: array-like, shape=[n_sample, n, n]

        Returns
        ------
        basis_representation: array-like, shape=[n_sample, dimension]
        """
        raise NotImplementedError("basis_representation not implemented.")

    def matrix_representation(self, basis_representation):
        """
        Parameters
        ----------
        basis_representation: array-like, shape=[n_sample, dimension]

        Returns
        ------
        matrix_representation: array-like, shape=[n_sample, n, n]
        """
        basis_representation = gs.to_ndarray(basis_representation, to_ndim=2)

        if self.basis is None:
            raise NotImplementedError("basis not implemented")

        return gs.einsum("ni,ijk ->njk", basis_representation, self.basis)
