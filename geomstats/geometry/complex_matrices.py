"""Module exposing the `ComplexMatrices` class.

Lead author: Yann Cabanes.
"""

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.base import ComplexVectorSpace
from geomstats.geometry.hermitian import HermitianMetric
from geomstats.geometry.matrices import Matrices, MatricesMetric


class ComplexMatrices(ComplexVectorSpace):
    """Class for the space of complex matrices (m, n).

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    """

    def __init__(self, m, n, **kwargs):
        geomstats.errors.check_integer(n, "n")
        geomstats.errors.check_integer(m, "m")
        kwargs.setdefault("metric", ComplexMatricesMetric(m, n))
        kwargs.setdefault("default_point_type", "matrix")
        super(ComplexMatrices, self).__init__(shape=(m, n), **kwargs)
        self.m = m
        self.n = n

    def _create_basis(self):
        """Create the canonical basis."""
        cdtype = gs.get_default_cdtype()
        m, n = self.m, self.n
        basis = gs.reshape(gs.eye(n * m, dtype=cdtype), (n * m, m, n))
        return basis

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
        is_matrix = super().belongs(point, atol=gs.atol)
        belongs = gs.logical_and(is_matrix, gs.is_complex(point))
        return belongs

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
        return Matrices.equal(mat_a, mat_b, atol=atol)

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
        return Matrices.mul(*args)

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
        return Matrices.bracket(mat_a, mat_b)

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
        return Matrices.transpose(mat)

    @staticmethod
    def transconjugate(mat):
        """Return the transconjugate of matrices.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        transconjugate : array-like, shape=[..., n, n]
            Transconjugated matrix.
        """
        is_vectorized = gs.ndim(gs.array(mat)) == 3
        axes = (0, 2, 1) if is_vectorized else (1, 0)
        return gs.transpose(gs.conj(mat), axes)

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
        return Matrices.diagonal(mat)

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
        return Matrices.is_square(mat)

    @classmethod
    def is_diagonal(cls, mat, atol=gs.atol):
        """Check if a matrix is square and diagonal.

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
        return Matrices.is_diagonal(mat, atol=atol)

    @classmethod
    def is_lower_triangular(cls, mat, atol=gs.atol):
        """Check if a matrix is lower triangular.

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
        return Matrices.is_lower_triangular(mat, atol=atol)

    @classmethod
    def is_upper_triangular(cls, mat, atol=gs.atol):
        """Check if a matrix is upper triangular.

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
        return Matrices.is_upper_triangular(mat, atol=atol)

    @classmethod
    def is_strictly_lower_triangular(cls, mat, atol=gs.atol):
        """Check if a matrix is strictly lower triangular.

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
        return Matrices.is_strictly_lower_triangular(mat, atol=atol)

    @classmethod
    def is_strictly_upper_triangular(cls, mat, atol=gs.atol):
        """Check if a matrix is strictly upper triangular.

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
        return Matrices.is_strictly_upper_triangular(mat, atol=atol)

    @classmethod
    def is_symmetric(cls, mat, atol=gs.atol):
        """Check if a matrix is symmetric.

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
        return Matrices.is_symmetric(mat, atol=gs.atol)

    @classmethod
    def is_hermitian(cls, mat, atol=gs.atol):
        """Check if a matrix is Hermitian.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_herm : array-like, shape=[...,]
            Boolean evaluating if the matrix is symmetric.
        """
        is_square = cls.is_square(mat)
        if not is_square:
            is_vectorized = gs.ndim(gs.array(mat)) == 3
            return gs.array([False] * len(mat)) if is_vectorized else False
        return cls.equal(mat, cls.transconjugate(mat), atol)

    @classmethod
    def is_pd(cls, mat):
        """Check if a matrix is positive definite.

        If the input matrix is complex,
        also check if the matrix is Hermitian.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        is_pd : array-like, shape=[...,]
            Boolean evaluating if the matrix is positive definite.
        """
        return Matrices.is_pd(mat)

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
        return Matrices.is_spd(mat, atol=atol)

    @classmethod
    def is_hpd(cls, mat, atol=gs.atol):
        """Check if a matrix is Hermitian positive definite.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_hpd : array-like, shape=[...,]
            Boolean evaluating if the matrix is Hermitian positive definite.
        """
        is_hpd = gs.logical_and(cls.is_hermitian(mat, atol), cls.is_pd(mat))
        return is_hpd

    @classmethod
    def is_skew_symmetric(cls, mat, atol=gs.atol):
        """Check if a matrix is skew-symmetric.

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
        return Matrices.is_skew_symmetric(mat, atol=atol)

    @classmethod
    def is_skew_hermitian(cls, mat, atol=gs.atol):
        """Check if a matrix is skew-Hermitian.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_skew_herm : array-like, shape=[...,]
            Boolean evaluating if the matrix is skew-Hermitian.
        """
        is_square = cls.is_square(mat)
        if not is_square:
            is_vectorized = gs.ndim(gs.array(mat)) == 3
            return gs.array([False] * len(mat)) if is_vectorized else False
        return cls.equal(mat, -cls.transconjugate(mat), atol)

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
        return Matrices.to_diagonal(mat)

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
        return Matrices.to_lower_triangular(mat)

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
        return Matrices.to_upper_triangular(mat)

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
        return Matrices.to_strictly_lower_triangular(mat)

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
        return Matrices.to_strictly_upper_triangular(mat)

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
        return Matrices.to_symmetric(mat)

    @classmethod
    def to_hermitian(cls, mat):
        """Make a matrix Hermitian.

        Make a matrix Hermitian by averaging it
        with its transconjugate.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        herm : array-like, shape=[..., n, n]
            Hermitian matrix.
        """
        return 1 / 2 * (mat + cls.transconjugate(mat))

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
        return Matrices.to_skew_symmetric(mat)

    @classmethod
    def to_skew_hermitian(cls, mat):
        """Make a matrix skew-Hermitian.

        Make matrix skew-Hermitian by averaging it
        with minus its transconjugate.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        skew_sym : array-like, shape=[..., n, n]
            Skew-Hermitian matrix.
        """
        return 1 / 2 * (mat - cls.transconjugate(mat))

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
        return Matrices.to_lower_triangular_diagonal_scaled(mat, K=K)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample from a uniform distribution in a complex cube.

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
        cdtype = gs.get_default_cdtype()

        m, n = self.m, self.n
        size = (n_samples, m, n) if n_samples != 1 else (m, n)
        point = gs.cast(bound * (gs.random.rand(*size) - 0.5), dtype=cdtype)
        point += 1j * gs.cast(bound * (gs.random.rand(*size) - 0.5), dtype=cdtype)
        return point

    def random_tangent_vec(self, base_point, n_samples=1):
        """Generate random tangent vec.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        base_point :  array-like, shape=[..., dim]
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vec at base point.
        """
        if (
            n_samples > 1
            and base_point.ndim > len(self.shape)
            and n_samples != len(base_point)
        ):
            raise ValueError(
                "The number of base points must be the same as the "
                "number of samples, when different from 1."
            )
        cdtype = gs.get_default_cdtype()

        tangent_vec = gs.cast(
            gs.random.normal(size=(n_samples,) + self.shape),
            dtype=cdtype,
        )
        tangent_vec += 1j * gs.cast(
            gs.random.normal(size=(n_samples,) + self.shape),
            dtype=cdtype,
        )
        return gs.squeeze(tangent_vec)

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
        return cls.mul(mat_2, mat_1, cls.transconjugate(mat_2))

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
        return gs.einsum("...ij,...ij->...", gs.conj(mat_1), mat_2)

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
        return Matrices.trace_product(mat_1, mat_2)

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
        return Matrices.flatten(self, mat)

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
        return Matrices.reshape(self, vec)

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
        return Matrices.align_matrices(point, base_point)


class ComplexMatricesMetric(HermitianMetric):
    """Hermitian metric on complex matrices given by Frobenius inner-product.

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    """

    def __init__(self, m, n, **kwargs):
        self.dim = m * n
        super(ComplexMatricesMetric, self).__init__(dim=self.dim, shape=(m, n))

    @staticmethod
    def inner_product(tangent_vec_a, tangent_vec_b, base_point=None):
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
        return ComplexMatrices.frobenius_product(tangent_vec_a, tangent_vec_b)

    def squared_norm(self, vector, base_point=None):
        """Compute the square of the norm of a vector.

        Squared norm of a vector associated to the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm.
        """
        sq_norm = self.inner_product(vector, vector, base_point)
        sq_norm = gs.real(sq_norm)
        return sq_norm

    @staticmethod
    def norm(vector, base_point=None):
        """Compute norm of a complex matrix.

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
        return MatricesMetric.norm(vector)
