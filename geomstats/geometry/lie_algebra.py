import geomstats.backend as gs


class MatrixLieAlgebra():
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
        bch = matrix_a + matrix_b + 0.5 * self.lie_brackets(matrix_a, matrix_b)
        return bch

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

        return gs.einsum('ni,ijk ->njk', basis_representation, self.basis)
