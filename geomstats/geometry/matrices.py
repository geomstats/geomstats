"""Module exposing the `Matrices` and `MatricesMetric` class."""
import logging
import math
from functools import reduce

import geomstats.backend as gs
import geomstats.errors
from geomstats.algebra_utils import flip_determinant, from_vector_to_diagonal_matrix
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.vectorization import repeat_out


class Matrices(VectorSpace):
    """Class for the space of matrices (m, n).

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    """

    def __init__(self, m, n, equip=True, equip_maximally=True):
        geomstats.errors.check_integer(n, "n")
        geomstats.errors.check_integer(m, "m")

        super().__init__(shape=(m, n), equip=equip, equip_maximally=equip_maximally)
        self.m = m
        self.n = n

    def _create_basis(self):
        """Create the canonical basis."""
        m, n = self.m, self.n
        return gs.reshape(gs.eye(n * m), (n * m, m, n))

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MatricesMetric

    @staticmethod
    def equal(mat_a, mat_b, atol=gs.atol):
        """Test if matrices a and b are close.

        Parameters
        ----------
        mat_a : array-like, shape=[..., dim1, dim2]
            Matrix.
        mat_b : array-like, shape=[..., dim2, dim3]
            Matrix.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        eq : array-like, shape=[...,]
            Boolean evaluating if the matrices are close.
        """
        return gs.all(gs.isclose(mat_a, mat_b, atol=atol), (-2, -1))

    @staticmethod
    def mul(*args):
        """Compute the product of matrices a1, ..., an.

        Parameters
        ----------
        a1 : array-like, shape=[..., dim_1, dim_2]
            Matrix.
        a2 : array-like, shape=[..., dim_2, dim_3]
            Matrix.
        ...
        an : array-like, shape=[..., dim_n-1, dim_n]
            Matrix.

        Returns
        -------
        mul : array-like, shape=[..., dim_1, dim_n]
            Result of the product of matrices.
        """
        return reduce(gs.matmul, args)

    @classmethod
    def bracket(cls, mat_a, mat_b):
        """Compute the commutator of a and b, i.e. `[a, b] = ab - ba`.

        Parameters
        ----------
        mat_a : array-like, shape=[..., n, n]
            Matrix.
        mat_b : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        mat_c : array-like, shape=[..., n, n]
            Commutator.
        """
        return cls.mul(mat_a, mat_b) - cls.mul(mat_b, mat_a)

    @staticmethod
    def transpose(mat):
        """Return the transpose of matrices.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        transpose : array-like, shape=[..., n, n]
            Transposed matrix.
        """
        ndim = gs.ndim(mat)
        axes = list(range(0, ndim))
        axes[-1] = ndim - 2
        axes[-2] = ndim - 1
        return gs.transpose(mat, axes)

    @staticmethod
    def diagonal(mat):
        """Return the diagonal of a matrix as a vector.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Matrix.

        Returns
        -------
        diagonal : array-like, shape=[..., min(m, n)]
            Vector of diagonal coefficients.
        """
        return gs.diagonal(mat, axis1=-2, axis2=-1)

    @staticmethod
    def is_square(mat):
        """Check if a matrix is square.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Matrix.

        Returns
        -------
        is_square : array-like, shape=[...,]
            Boolean evaluating if the matrix is square.
        """
        shape = mat.shape[:-2]
        if mat.shape[-1] == mat.shape[-2]:
            return gs.ones(shape, dtype=bool)

        return gs.zeros(shape, dtype=bool)

    @classmethod
    def is_diagonal(cls, mat, atol=gs.atol):
        """Check if a square matrix is diagonal.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_diagonal : array-like, shape=[...,]
            Boolean evaluating if the matrix is square and diagonal.
        """
        diagonal_mat = from_vector_to_diagonal_matrix(cls.diagonal(mat))
        is_diagonal = gs.all(gs.isclose(mat, diagonal_mat, atol=atol), axis=(-2, -1))
        return is_diagonal

    @classmethod
    def is_lower_triangular(cls, mat, atol=gs.atol):
        """Check if a square matrix is lower triangular.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default : backend atol.

        Returns
        -------
        is_tril : array-like, shape=[...,]
            Boolean evaluating if the matrix is lower triangular
        """
        return cls.equal(mat, gs.tril(mat), atol)

    @classmethod
    def is_upper_triangular(cls, mat, atol=gs.atol):
        """Check if a square matrix is upper triangular.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default : backend atol.

        Returns
        -------
        is_triu : array-like, shape=[...,]
            Boolean evaluating if the matrix is upper triangular.
        """
        return cls.equal(mat, gs.triu(mat), atol)

    @classmethod
    def is_strictly_lower_triangular(cls, mat, atol=gs.atol):
        """Check if a square matrix is strictly lower triangular.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default : backend atol.

        Returns
        -------
        is_strictly_tril : array-like, shape=[...,]
            Boolean evaluating if the matrix is strictly lower triangular
        """
        return cls.equal(mat, gs.tril(mat, k=-1), atol)

    @classmethod
    def is_strictly_upper_triangular(cls, mat, atol=gs.atol):
        """Check if a square matrix is strictly upper triangular.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default : backend atol.

        Returns
        -------
        is_strictly_triu : array-like, shape=[...,]
            Boolean evaluating if the matrix is strictly upper triangular
        """
        return cls.equal(mat, gs.triu(mat, k=1))

    @classmethod
    def is_symmetric(cls, mat, atol=gs.atol):
        """Check if a square matrix is symmetric.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_sym : array-like, shape=[...,]
            Boolean evaluating if the matrix is symmetric.
        """
        return cls.equal(mat, cls.transpose(mat), atol)

    @classmethod
    def is_pd(cls, mat):
        """Check if a matrix is positive definite.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        is_pd : array-like, shape=[...,]
            Boolean evaluating if the matrix is positive definite.
        """
        if mat.ndim == 2:
            return gs.array(gs.linalg.is_single_matrix_pd(mat))

        shape = mat.shape
        if mat.ndim > 3:
            mat = gs.reshape(mat, (-1,) + shape[-2:])

        is_pd = gs.array([gs.linalg.is_single_matrix_pd(m) for m in mat])
        return gs.reshape(is_pd, shape[:-2])

    @classmethod
    def is_spd(cls, mat, atol=gs.atol):
        """Check if a matrix is symmetric positive definite.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_spd : array-like, shape=[...,]
            Boolean evaluating if the matrix is symmetric positive definite.
        """
        return gs.logical_and(cls.is_symmetric(mat, atol), cls.is_pd(mat))

    @classmethod
    def is_skew_symmetric(cls, mat, atol=gs.atol):
        """Check if a square matrix is skew symmetric.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_skew_sym : array-like, shape=[...,]
            Boolean evaluating if the matrix is skew-symmetric.
        """
        return cls.equal(mat, -cls.transpose(mat), atol)

    @classmethod
    def to_diagonal(cls, mat):
        """Make a matrix diagonal.

        Make a matrix diagonal by zeroing out non
        diagonal elements.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        diag : array-like, shape=[..., n, n]
        """
        return cls.to_upper_triangular(cls.to_lower_triangular(mat))

    @classmethod
    def to_lower_triangular(cls, mat):
        """Make a matrix lower triangular.

        Make a matrix lower triangular by zeroing
        out upper elements.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        tril : array-like, shape=[..., n, n]
            Lower  triangular matrix.
        """
        return gs.tril(mat)

    @classmethod
    def to_upper_triangular(cls, mat):
        """Make a matrix upper triangular.

        Make a matrix upper triangular by zeroing
        out lower elements.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        triu : array-like, shape=[..., n, n]
        """
        return gs.triu(mat)

    @classmethod
    def to_strictly_lower_triangular(cls, mat):
        """Make a matrix strictly lower triangular.

        Make a matrix stricly lower triangular by zeroing
        out upper and diagonal elements.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        tril : array-like, shape=[..., n, n]
            Lower  triangular matrix.
        """
        return gs.tril(mat, k=-1)

    @classmethod
    def to_strictly_upper_triangular(cls, mat):
        """Make a matrix stritcly upper triangular.

        Make a matrix strictly upper triangular by zeroing
        out lower and diagonal elements.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        triu : array-like, shape=[..., n, n]
        """
        return gs.triu(mat, k=1)

    @classmethod
    def to_symmetric(cls, mat):
        """Make a matrix symmetric.

        Make a matrix symmetric by averaging it
        with its transpose.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        sym : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        return 1 / 2 * (mat + cls.transpose(mat))

    @classmethod
    def to_skew_symmetric(cls, mat):
        """Make a matrix skew-symmetric.

        Make matrix skew-symmetric by averaging it
        with minus its transpose.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        skew_sym : array-like, shape=[..., n, n]
            Skew-symmetric matrix.
        """
        return 1 / 2 * (mat - cls.transpose(mat))

    @classmethod
    def to_lower_triangular_diagonal_scaled(cls, mat, K=2.0):
        """Make a matrix lower triangular.

        Make matrix lower triangular by zeroing out
        upper elements and divide diagonal by factor K.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        tril : array-like, shape=[..., n, n]
            Lower  triangular matrix.
        """
        slt = cls.to_strictly_lower_triangular(mat)
        diag = cls.to_diagonal(mat) / K
        return slt + diag

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

    @classmethod
    def congruent(cls, mat_1, mat_2):
        r"""Compute the congruent action of mat_2 on mat_1.

        This is :math:`mat\_2 \ mat\_1 \ mat\_2^T`.

        Parameters
        ----------
        mat_1 : array-like, shape=[..., n, n]
            Matrix.
        mat_2 : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        cong : array-like, shape=[..., n, n]
            Result of the congruent action.
        """
        return cls.mul(mat_2, mat_1, cls.transpose(mat_2))

    @staticmethod
    def frobenius_product(mat_1, mat_2):
        """Compute Frobenius inner-product of two matrices.

        The `einsum` function is used to avoid computing a matrix product. It
        is also faster than using a sum an element-wise product.

        Parameters
        ----------
        mat_1 : array-like, shape=[..., m, n]
            Matrix.
        mat_2 : array-like, shape=[..., m, n]
            Matrix.

        Returns
        -------
        product : array-like, shape=[...,]
            Frobenius inner-product of mat_1 and mat_2
        """
        return gs.einsum("...ij,...ij->...", mat_1, mat_2)

    @staticmethod
    def trace_product(mat_1, mat_2):
        """Compute trace of the product of two matrices.

        The `einsum` function is used to avoid computing a matrix product. It
        is also faster than using a sum an element-wise product.

        Parameters
        ----------
        mat_1 : array-like, shape=[..., m, n]
            Matrix.
        mat_2 : array-like, shape=[..., m, n]
            Matrix.

        Returns
        -------
        product : array-like, shape=[...,]
            Frobenius inner-product of mat_1 and mat_2
        """
        return gs.einsum("...ij,...ji->...", mat_1, mat_2)

    @staticmethod
    def flatten(mat):
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
        batch_shape = mat.shape[:-2]
        mat_shape = mat.shape[-2:]
        return gs.reshape(mat, batch_shape + (math.prod(mat_shape),))

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

    @classmethod
    def align_matrices(cls, point, base_point):
        """Align matrices.

        Find the optimal rotation R in SO(m) such that the base point and
        R.point are well positioned.

        Parameters
        ----------
        point : array-like, shape=[..., m, n]
            Point on the manifold.
        base_point : array-like, shape=[..., m, n]
            Point on the manifold.

        Returns
        -------
        aligned : array-like, shape=[..., m, n]
            R.point.
        """
        mat = gs.matmul(cls.transpose(point), base_point)
        left, singular_values, right = gs.linalg.svd(mat, full_matrices=False)
        det = gs.linalg.det(mat)
        conditioning = (
            singular_values[..., -2] + gs.sign(det) * singular_values[..., -1]
        ) / singular_values[..., 0]
        if gs.any(conditioning < gs.atol):
            logging.warning(
                f"Singularity close, ill-conditioned matrix "
                f"encountered: "
                f"{conditioning[conditioning < 1e-10]}"
            )
        if gs.any(gs.isclose(conditioning, 0.0)):
            logging.warning("Alignment matrix is not unique.")
        flipped = flip_determinant(cls.transpose(right), det)
        return Matrices.mul(point, left, cls.transpose(flipped))


class MatricesMetric(EuclideanMetric):
    """Euclidean metric on matrices given by Frobenius inner-product."""

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
        inner_prod = Matrices.frobenius_product(tangent_vec_a, tangent_vec_b)
        return repeat_out(
            self._space.point_ndim, inner_prod, tangent_vec_a, tangent_vec_b, base_point
        )

    def norm(self, vector, base_point=None):
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
        norm = gs.linalg.norm(vector, axis=(-2, -1))
        return repeat_out(self._space.point_ndim, norm, vector, base_point)
