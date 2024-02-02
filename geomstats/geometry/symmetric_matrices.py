"""The vector space of symmetric matrices.

Lead author: Yann Thanwerdas.
"""

import geomstats.backend as gs
from geomstats.geometry.base import LevelSet, MatrixVectorSpace
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import repeat_out


class SymmetricMatrices(MatrixVectorSpace):
    """Class for the vector space of symmetric matrices of size n.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.
    """

    def __init__(self, n, equip=True):
        super().__init__(dim=int(n * (n + 1) / 2), shape=(n, n), equip=equip)
        self.n = n

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MatricesMetric

    def _create_basis(self):
        """Compute the basis of the vector space of symmetric matrices."""
        indices, values = [], []
        k = -1
        for row in range(self.n):
            for col in range(row, self.n):
                k += 1
                if row == col:
                    indices.append((k, row, row))
                    values.append(1.0)
                else:
                    indices.extend([(k, row, col), (k, col, row)])
                    values.extend([1.0, 1.0])

        return gs.array_from_sparse(indices, values, (k + 1, self.n, self.n))

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a matrix is symmetric.

        Parameters
        ----------
        point : array-like, shape=[.., n, n]
            Point to test.
        atol : float
            Tolerance to evaluate equality with the transpose.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        belongs = super().belongs(point)
        if gs.any(belongs):
            is_symmetric = Matrices.is_symmetric(point, atol)
            return gs.logical_and(belongs, is_symmetric)
        return belongs

    def projection(self, point):
        """Make a matrix symmetric, by averaging with its transpose.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        sym : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        return Matrices.to_symmetric(point)

    @staticmethod
    def basis_representation(matrix_representation):
        """Convert a symmetric matrix into a vector.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., n(n+1)/2]
            Vector.
        """
        return gs.triu_to_vec(matrix_representation)

    @staticmethod
    def matrix_representation(basis_representation):
        """Convert a vector into a symmetric matrix.

        Parameters
        ----------
        basis_representation : array-like, shape=[..., n(n+1)/2]
            Vector.

        Returns
        -------
        matrix_representation : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        vec_dim = basis_representation.shape[-1]
        mat_dim = (gs.sqrt(8.0 * vec_dim + 1) - 1) / 2
        if mat_dim != int(mat_dim):
            raise ValueError(
                "Invalid input dimension, it must be of the form"
                "(n_samples, n * (n + 1) / 2)"
            )
        mat_dim = int(mat_dim)
        shape = (mat_dim, mat_dim)
        mask = 2 * gs.ones(shape) - gs.eye(mat_dim)
        indices = list(zip(*gs.triu_indices(mat_dim)))
        if gs.ndim(basis_representation) == 1:
            upper_triangular = gs.array_from_sparse(
                indices, basis_representation, shape
            )
        else:
            upper_triangular = gs.stack(
                [
                    gs.array_from_sparse(indices, data, shape)
                    for data in basis_representation
                ]
            )

        mat = Matrices.to_symmetric(upper_triangular) * mask
        return mat


class SymmetricHollowMatrices(MatrixVectorSpace, LevelSet):
    """Space of symmetric hollow matrices."""

    def __init__(self, n, equip=True):
        self.n = n
        super().__init__(dim=int(n * (n - 1) / 2), shape=(n, n), equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MatricesMetric

    def _define_embedding_space(self):
        return SymmetricMatrices(n=self.n)

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]

        Returns
        -------
        submersed_point : array-like, shape=[..., n]
        """
        return Matrices.diagonal(point)

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
        point : Ignored.

        Returns
        -------
        submersed_vector : array-like, shape=[..., n]
        """
        out = self.submersion(vector)
        return repeat_out(self.point_ndim, out, vector, point, out_shape=(self.n,))

    def _create_basis(self):
        """Compute the basis of the vector space of hollow symmetric matrices."""
        indices, values = [], []
        k = -1
        for row in range(self.n):
            for col in range(row + 1, self.n):
                k += 1
                indices.extend([(k, row, col), (k, col, row)])
                values.extend([1.0, 1.0])

        return gs.array_from_sparse(indices, values, (k + 1, self.n, self.n))

    @staticmethod
    def basis_representation(matrix_representation):
        """Convert a hollow symmetric matrix into a vector.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        vec : array-like, shape=[..., n(n+1)/2]
            Vector.
        """
        return gs.tril_to_vec(matrix_representation, k=-1)

    def projection(self, point):
        """Project a point in embedding manifold on embedded manifold.

        Parameters
        ----------
        point : array-like, shape=[..., *embedding_space.point_shape]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., *point_shape]
            Projected point.
        """
        return point - Matrices.to_diagonal(point)


class HollowMatricesPermutationInvariantMetric(RiemannianMetric):
    """A permutation-invariant metric on the space of hollow matrices.

    It is flat Riemannian metric à priori invariant by the congruence action
    of permutation matrices defined over a matrix vector space,
    so it is just a metric defined over the vector space itself,
    even though we implement it as a Riemannian metric, since
    the tangent bundle to Hol is itself, so tangent vector are simply hollow matrices,
    and since the metric is flat it doesn't depend
    from the base point.
    """

    def __init__(self, space, alpha=1.0, beta=1.0, gamma=1.0):
        # the condition is always verified for for a,b,g=1 (easy proof) <=> n>0
        # TODO: add condition check?
        self._check_params(space, alpha, beta, gamma)
        super().__init__(space=space)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _check_params(space, alpha, beta, gamma):
        if space.n == 2:
            if alpha < gs.atol or beta < gs.atol:
                raise ValueError("{alpha} and {beta} must be 0. when n==2")

        elif space.n == 3:
            if alpha < gs.atol:
                raise ValueError("{alpha} must be 0. when n==2")

    def quadratic_form(self, tangent_vec):
        comp = tangent_vec @ tangent_vec

        out_alpha = self.alpha * gs.trace(comp) if self.alpha > gs.atol else 0.0
        out_beta = self.beta * gs.sum(comp) if self.beta > gs.atol else 0.0
        out_gamma = self.gamma * gs.sum(tangent_vec) ** 2

        return out_alpha + out_beta + out_gamma

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        return (1 / 2) * (
            self.quadratic_form(tangent_vec_a + tangent_vec_b)
            - self.quadratic_form(tangent_vec_a)
            - self.quadratic_form(tangent_vec_b)
        )

    def exp(self, tangent_vec, base_point):
        """Compute exp map of a base point in tangent vector direction.

        The Riemannian exponential is vector addition in the Euclidean space.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Riemannian exponential.
        """
        return base_point + tangent_vec

    def log(self, point, base_point):
        """Compute log map using a base point and other point.

        The Riemannian logarithm is the subtraction in the Euclidean space.

        Parameters
        ----------
        point: array-like, shape=[..., dim]
            Point.
        base_point: array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        log: array-like, shape=[..., dim]
            Riemannian logarithm.
        """
        return point - base_point
