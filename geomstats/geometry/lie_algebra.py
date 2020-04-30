"""Module providing an implementation of MatrixLieAlgebras.

There are two main forms of representation for elements of a MatrixLieAlgebra
implemented here. The first one is as a matrix, as elements of R^(n x n).
The second is by choosing a base and remembering the coefficients of an element
in that base. This base will be provided in child classes
(e.g. SkewSymmetricMatrices).
"""
import geomstats.backend as gs
import geomstats.errors
from ._bch_coefficients import BCH_COEFFICIENTS


class MatrixLieAlgebra:
    """Class implementing matrix Lie algebra related functions.

    Parameters
    ----------
    dim : int
        Dimension of the Lie algebra as a real vector space.
    n : int
        Amount of rows and columns in the matrix representation of the
        Lie algebra.
    """

    def __init__(self, dim, n):
        geomstats.errors.check_integer(dim, 'dim')
        geomstats.errors.check_integer(n, 'n')
        self.dim = dim
        self.n = n
        self.basis = None

    @staticmethod
    def lie_bracket(matrix_a, matrix_b):
        """Compute the Lie_bracket (commutator) of two matrices.

        Notice that inputs have to be given in matrix form, no conversion
        between basis and matrix representation is attempted.

        Parameters
        ----------
        matrix_a : array-like, shape=[..., n, n]
            Matrix.
        matrix_b : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        bracket : shape=[..., n, n]
            Lie bracket.
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
        matrix_a, matrix_b : array-like, shape=[..., n, n]
            Matrices.
        order : int
            The order to which the approximation is calculated. Note that this
            is NOT the same as using only e_i with i < order.
            Optional, default 2.

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

        el = [matrix_a, matrix_b]
        result = matrix_a + matrix_b

        for i in gs.arange(2, n_terms):
            i_p = BCH_COEFFICIENTS[i, 1] - 1
            i_pp = BCH_COEFFICIENTS[i, 2] - 1

            el.append(self.lie_bracket(el[i_p], el[i_pp]))
            result += (float(BCH_COEFFICIENTS[i, 3]) /
                       float(BCH_COEFFICIENTS[i, 4]) *
                       el[i])
        return result

    def basis_representation(self, matrix_representation):
        """Compute the coefficients of matrices in the given basis.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis.
        """
        raise NotImplementedError("basis_representation not implemented.")

    def matrix_representation(self, basis_representation):
        """Compute the matrix representation for the given basis coefficients.

        Sums the basis elements according to the coefficents given in
        basis_representation.

        Parameters
        ----------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis.

        Returns
        -------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.
        """
        basis_representation = gs.to_ndarray(basis_representation, to_ndim=2)

        if self.basis is None:
            raise NotImplementedError("basis not implemented")

        return gs.einsum("ni,ijk ->njk", basis_representation, self.basis)
