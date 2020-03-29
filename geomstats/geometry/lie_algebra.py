"""Module providing an implementation of MatrixLieAlgebras.

There are two main forms of representation for elements of a MatrixLieAlgebra
implemented here. The first one is as a matrix, as elements of R^(n x n).
The second is by choosing a base and remembering the coefficients of an element
in that base. This base will be provided in child classes
(e.g. SkewSymmetricMatrices).
"""
import geomstats.backend as gs
from ._bch_coefficients import BCH_COEFFICIENTS


class MatrixLieAlgebra:
    """Class implementing matrix Lie algebra related functions."""

    def __init__(self, dimension, n):
        """Construct the MatrixLieAlgebra object.

        Parameters
        ----------
        dimension: int
            The dimension of the Lie algebra as a real vector space
        n: int
            The amount of rows and columns in the matrx representation of the
            Lie algebra
        """
        self.dimension = dimension
        self.n = n
        self.basis = None

    def lie_bracket(self, matrix_a, matrix_b):
        """Compute the Lie_bracket (commutator) of two matrices.

        Notice that inputs have to be given in matrix form, no conversion
        between basis and matrix representation is attempted.

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
        """Calculate the Baker-Campbell-Hausdorff approximation of given order.

        The implementation is based on [CM2009a]_ with the pre-computed
        constants taken from [CM2009b]_. Our coefficients are truncated to
        enable us to calculate BCH up to order 15.

        This represents Z = log(exp(X)exp(Y)) as an infinite linear combination
        of the form Z = sum z_i e_i where z_i are rational numbers and e_i are
        iterated Lie brackets starting with e_1 = X, e_2 = Y, each e_i is given
        by some i',i'': e_i = [e_i', e_i''].

        Parameters
        ----------
        matrix_a, matrix_b : array-like, shape=[n_sample, n, n]
        order : int
            The order to which the approximation is calculated. Note that this
            is NOT the same as using only e_i with i < order.

        References
        ----------
        .. [CM2009a] F. Casas and A. Murua. An efficient algorithm for
           computing the Baker–Campbell–Hausdorff series and some of its
           applications. Journal of Mathematical Physics 50, 2009
        .. [CM2009b] http://www.ehu.eus/ccwmuura/research/bchHall20.dat
        """
        if order > 15:
            raise NotImplementedError("BCH is not implemented for order > 15.")

        number_of_hom_degree = gs.array(
            [2, 1, 2, 3, 6, 9, 18, 30, 56, 99, 186, 335, 630, 1161, 2182])
        n_terms = gs.sum(number_of_hom_degree[:order])

        ei = gs.zeros((n_terms, self.n, self.n))
        ei[0] = matrix_a
        ei[1] = matrix_b
        result = matrix_a + matrix_b

        for i in gs.arange(2, n_terms):
            i_p = BCH_COEFFICIENTS[i, 1] - 1
            i_pp = BCH_COEFFICIENTS[i, 2] - 1

            ei[i] = self.lie_bracket(ei[i_p], ei[i_pp])
            result += (BCH_COEFFICIENTS[i, 3] /
                       float(BCH_COEFFICIENTS[i, 4]) *
                       ei[i])

        return result

    def basis_representation(self, matrix_representation):
        """Compute the coefficients of matrices in the given base.

        Parameters
        ----------
        matrix_representation: array-like, shape=[n_sample, n, n]

        Returns
        -------
        basis_representation: array-like, shape=[n_sample, dimension]
        """
        raise NotImplementedError("basis_representation not implemented.")

    def matrix_representation(self, basis_representation):
        """Compute the matrix representation for the given basis coefficients.

        Sums the basis elements according to the coefficents given in
        basis_representation.

        Parameters
        ----------
        basis_representation: array-like, shape=[n_sample, dimension]

        Returns
        -------
        matrix_representation: array-like, shape=[n_sample, n, n]
        """
        basis_representation = gs.to_ndarray(basis_representation, to_ndim=2)

        if self.basis is None:
            raise NotImplementedError("basis not implemented")

        return gs.einsum("ni,ijk ->njk", basis_representation, self.basis)
