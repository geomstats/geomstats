"""Module exposing the `Matrices` and `MatricesMetric` class."""

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.euclidean import EuclideanMetric


class Matrices(VectorSpace):
    """Class for the space of matrices (m, n).

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    """

    def __init__(self, m, n, **kwargs):
        geomstats.errors.check_integer(n, "n")
        geomstats.errors.check_integer(m, "m")
        kwargs.setdefault("metric", MatricesMetric(m, n))
        super().__init__(shape=(m, n), **kwargs)
        self.m = m
        self.n = n

    def _create_basis(self):
        """Create the canonical basis."""
        m, n = self.m, self.n
        return gs.reshape(gs.eye(n * m), (n * m, m, n))

    def belongs(self, point, atol=gs.atol):
        """Check if point belongs to the Matrices space.

        Parameters
        ----------
        point : array-like, shape=[..., m, n]
            Point to be checked.
        atol : float
            Unused here.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the Matrices space.
        """
        ndim = point.ndim
        if ndim == 1:
            return False
        mat_dim_1, mat_dim_2 = point.shape[-2:]
        belongs = (mat_dim_1 == self.m) and (mat_dim_2 == self.n)
        return belongs if ndim == 2 else gs.tile(gs.array([belongs]), [point.shape[0]])

    def random_point(self, n_samples=1, bound=1.0):
        """Sample from a uniform distribution in a cube.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample each entry.
            Optional, default: 1.

        Returns
        -------
        point : array-like, shape=[..., m, n]
            Sample.
        """
        m, n = self.m, self.n
        size = (n_samples, m, n) if n_samples != 1 else (m, n)
        point = bound * (gs.random.rand(*size) - 0.5)
        return point

    def flatten(self, mat):
        """Return a flattened form of the matrix.

        Flatten a matrix (compatible with vectorization on data axis 0).
        The reverse operation is reshape. These operations are often called
        matrix vectorization / matricization in mathematics
        (https://en.wikipedia.org/wiki/Tensor_reshaping).
        The names flatten / reshape were chosen to avoid  confusion with
        vectorization on data axis 0.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Matrix.

        Returns
        -------
        vec : array-like, shape=[..., m * n]
            Flatten copy of mat.
        """
        is_data_vectorized = gs.ndim(gs.array(mat)) == 3
        shape = (
            (mat.shape[0], self.m * self.n)
            if is_data_vectorized
            else (self.m * self.n,)
        )
        return gs.reshape(mat, shape)

    def reshape(self, vec):
        """Return a matricized form of the vector.

        Matricize a vector (compatible with vectorization on data axis 0).
        The reverse operation is matrices.flatten. These operations are often
        called matrix vectorization / matricization in mathematics
        (https://en.wikipedia.org/wiki/Tensor_reshaping).
        The names flatten / reshape were chosen to avoid  confusion with
        vectorization on data axis 0.

        Parameters
        ----------
        vec : array-like, shape=[..., m * n]
            Vector.

        Returns
        -------
        mat : array-like, shape=[..., m, n]
            Matricized copy of vec.
        """
        is_data_vectorized_on_axis_0 = gs.ndim(gs.array(vec)) == 2
        if is_data_vectorized_on_axis_0:
            vector_size = vec.shape[1]
            shape = (vec.shape[0], self.m, self.n)
        else:
            vector_size = vec.shape[0]
            shape = (
                self.m,
                self.n,
            )

        if vector_size != self.m * self.n:
            raise ValueError("Incompatible vector and matrix sizes")
        return gs.reshape(vec, shape)


class MatricesMetric(EuclideanMetric):
    """Euclidean metric on matrices given by Frobenius inner-product.

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    """

    def __init__(self, m, n, **kwargs):
        dimension = m * n
        super().__init__(dim=dimension, shape=(m, n))

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute Frobenius inner-product of two tangent vectors.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., m, n]
            Tangent vector.
        tangent_vec_b : array-like, shape=[..., m, n]
            Tangent vector.
        base_point : array-like, shape=[..., m, n]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Frobenius inner-product of tangent_vec_a and tangent_vec_b.
        """
        return gs.matrices.frobenius_product(tangent_vec_a, tangent_vec_b)

    @staticmethod
    def norm(vector, base_point=None):
        """Compute norm of a matrix.

        Norm of a matrix associated to the Frobenius inner product.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        norm : array-like, shape=[...,]
            Norm.
        """
        return gs.matrices.norm(vector)
