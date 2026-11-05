"""Stiefel manifold St(n,p).

A set of all orthonormal p-frames in n-dimensional space, where p <= n.

Lead author: Oleg Kachan.
"""

import warnings

import geomstats.backend as gs
import geomstats.errors
from geomstats import algebra_utils
from geomstats.geometry.base import LevelSet
from geomstats.geometry.hermitian_matrices import powermh
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.numerics.geodesic import LogSolver
from geomstats.vectorization import repeat_out


def _lower_right_block(matrix, p):
    """Extract the block after the first p rows and columns.

    Parameters
    ----------
    matrix : array-like, shape=[..., m, n]
        Matrix to extract the block from.
    p : int
        Number of leading rows and columns to skip.

    Returns
    -------
    block : array-like, shape=[..., m - p, n - p]
        Lower-right block.
    """
    return matrix[..., p:, p:]


def _lower_left_block(matrix, p):
    """Extract the block after the first p rows and within the first p columns.

    Parameters
    ----------
    matrix : array-like, shape=[..., m, n]
        Matrix to extract the block from.
    p : int
        Number of leading rows and columns defining the partition.

    Returns
    -------
    block : array-like, shape=[..., m - p, p]
        Lower-left block.
    """
    return matrix[..., p:, :p]


def _left_block(matrix, p):
    """Extract the first p columns.

    Parameters
    ----------
    matrix : array-like, shape=[..., m, n]
        Matrix to extract the block from.
    p : int
        Number of leading columns to extract.

    Returns
    -------
    block : array-like, shape=[..., m, p]
        Left block.
    """
    return matrix[..., :, :p]


def _right_block(matrix, p):
    """Extract the columns after the first p columns.

    Parameters
    ----------
    matrix : array-like, shape=[..., m, n]
        Matrix to extract the block from.
    p : int
        Number of leading columns to skip.

    Returns
    -------
    block : array-like, shape=[..., m, n - p]
        Right block.
    """
    return matrix[..., :, p:]


def _concatenate_columns(*matrices):
    """Concatenate matrices along the column axis.

    Parameters
    ----------
    *matrices : array-like, shape=[..., m, n_i]
        Matrices with matching batch shapes and numbers of rows.

    Returns
    -------
    matrix : array-like, shape=[..., m, sum(n_i)]
        Horizontally concatenated matrices.
    """
    return gs.concatenate(matrices, axis=-1)


def _concatenate_rows(*matrices):
    """Concatenate matrices along the row axis.

    Parameters
    ----------
    *matrices : array-like, shape=[..., m_i, n]
        Matrices with matching batch shapes and numbers of columns.

    Returns
    -------
    matrix : array-like, shape=[..., sum(m_i), n]
        Vertically concatenated matrices.
    """
    return gs.concatenate(matrices, axis=-2)


def _upper_left_block(matrix, p):
    """Extract the block within the first p rows and columns.

    Parameters
    ----------
    matrix : array-like, shape=[..., m, n]
        Matrix to extract the block from.
    p : int
        Number of leading rows and columns defining the partition.

    Returns
    -------
    block : array-like, shape=[..., p, p]
        Upper-left block.
    """
    return matrix[..., :p, :p]


class Stiefel(LevelSet):
    """Class for Stiefel manifolds St(n,p).

    A set of all orthonormal p-frames in n-dimensional space,
    where p <= n.

    Parameters
    ----------
    n : int
        Dimension of the ambient vector space.
    p : int
        Number of basis vectors in the orthonormal frame.
    """

    def __init__(self, n, p, equip=True):
        geomstats.errors.check_integer(n, "n")
        geomstats.errors.check_integer(p, "p")
        if p > n:
            raise ValueError("p needs to be smaller than n.")

        self.n = n
        self.p = p
        self._value = gs.eye(p)

        dim = int(p * n - (p * (p + 1) / 2))
        super().__init__(dim=dim, equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return StiefelCanonicalMetric

    def _define_embedding_space(self):
        return Matrices(self.n, self.p)

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n, p]

        Returns
        -------
        submersed_point : array-like, shape=[..., n, p]
        """
        return Matrices.mul(gs.transpose(point), point) - self._value

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., n, p]
        point : array-like, shape=[..., n, p]

        Returns
        -------
        submersed_vector : array-like, shape=[..., n, p]
        """
        return 2 * Matrices.to_symmetric(Matrices.mul(gs.transpose(point), vector))

    @staticmethod
    def to_grassmannian(point):
        r"""Project a point of St(n, p) to Gr(n, p).

        If :math:`U \in St(n, p)` is an orthonormal frame,
        return the orthogonal projector :math:`P = U U^T`
        onto the subspace of :math:`\mathbb{R}^n` spanned by
        :math:`U`.

        Parameters
        ----------
        point : array-like, shape=[..., n, p]
            Point.

        Returns
        -------
        projected : array-like, shape=[..., n, n]
            Projected point.
        """
        return Matrices.mul(point, gs.transpose(point))

    def random_uniform(self, n_samples=1):
        r"""Sample on St(n,p) from the uniform distribution.

        If :math:`Z(p,n) \sim N(0,1)`, then :math:`St(n,p) \sim U`,
        according to Haar measure:
        :math:`St(n,p) := Z(Z^TZ)^{-1/2}`.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n, p]
            Samples on the Stiefel manifold.
        """
        n, p = self.n, self.p
        size = (n_samples, n, p) if n_samples != 1 else (n, p)

        std_normal = gs.random.normal(size=size)
        std_normal_transpose = gs.transpose(std_normal)
        aux = Matrices.mul(std_normal_transpose, std_normal)
        inv_sqrt_aux = powermh(aux, -1.0 / 2)
        samples = Matrices.mul(std_normal, inv_sqrt_aux)

        return samples

    def random_point(self, n_samples=1, bound=1.0):
        r"""Sample on St(n,p) from the uniform distribution.

        If :math:`Z(p,n) \sim N(0,1)`, then :math:`St(n,p) \sim U`,
        according to Haar measure:
        :math:`St(n,p) := Z(Z^TZ)^{-1/2}`.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Unused here.

        Returns
        -------
        samples : array-like, shape=[..., n, p]
            Samples on the Stiefel manifold.
        """
        return self.random_uniform(n_samples)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Inspired by the method of Pymanopt.

        Parameters
        ----------
        vector : array-like, shape=[..., n, p]
            Vector.
        base_point : array-like, shape=[..., n, p]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, p]
            Tangent vector at base point.
        """
        aux = Matrices.mul(gs.transpose(base_point), vector)
        sym_aux = Matrices.to_symmetric(aux)
        return vector - Matrices.mul(base_point, sym_aux)

    def projection(self, point):
        """Project a close enough matrix to the Stiefel manifold.

        A singular value decomposition is used, and all singular values are
        set to 1 [AM2008]_

        Parameters
        ----------
        point : array-like, shape=[..., n, p]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., n, p]

        References
        ----------
        .. [AM2008] Absil, Pierre-Antoine, and Jérôme Malick.
            “Projection-like Retractions on Matrix Manifolds.”
            SIAM Journal on Optimization 22, no. 1 (January 2012):
            135–58. https://doi.org/10.1137/100802529.
        """
        p = self.p
        mat_u, _, mat_v = gs.linalg.svd(point)
        return Matrices.mul(_left_block(mat_u, p), mat_v)


class StiefelCanonicalMetric(RiemannianMetric):
    """Class that defines the canonical metric for Stiefel manifolds."""

    def __init__(self, space, log_solver_kwargs=None):
        super().__init__(space=space, signature=(space.dim, 0, 0))
        self.log_solver = _ZimmermannLogSolver(space, **(log_solver_kwargs or {}))

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the inner-product of two tangent vectors at a base point.

        Canonical inner-product on the tangent space at `base_point`,
        which is different from the inner-product induced by the embedding
        (see [RLSMRZ2017]_).

        .. math::

            \langle\Delta, \tilde{\Delta}\rangle_{U}=\operatorname{tr}
            \left(\Delta^{T}\left(I-\frac{1}{2} U U^{T}\right)
            \tilde{\Delta}\right)

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, p]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, p]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.

        Returns
        -------
        inner_prod : array-like, shape=[..., 1]
            Inner-product of the two tangent vectors.

        References
        ----------
        .. [RLSMRZ2017] R Zimmermann. A matrix-algebraic algorithm for the
            Riemannian logarithm on the Stiefel manifold under the canonical
            metric. SIAM Journal on Matrix Analysis and Applications 38 (2),
            322-342, 2017. https://epubs.siam.org/doi/pdf/10.1137/16M1074485

        """
        base_point_transpose = gs.transpose(base_point)

        identity = gs.eye(self._space.n)
        aux = gs.matmul(
            gs.transpose(tangent_vec_a),
            identity - 0.5 * gs.matmul(base_point, base_point_transpose),
        )
        return Matrices.trace_product(aux, tangent_vec_b)

    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, p]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.

        Returns
        -------
        exp : array-like, shape=[..., n, p]
            Point in the Stiefel manifold equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        p = self._space.p
        matrix_a = Matrices.mul(gs.transpose(base_point), tangent_vec)
        matrix_k = tangent_vec - Matrices.mul(base_point, matrix_a)

        matrix_q, matrix_r = gs.linalg.qr(matrix_k)

        matrix_ar = _concatenate_columns(matrix_a, -gs.transpose(matrix_r))

        matrix_rz = _concatenate_columns(matrix_r, gs.zeros_like(matrix_a))
        block = _concatenate_rows(matrix_ar, matrix_rz)
        matrix_mn_e = gs.linalg.expm(block)

        exp = Matrices.mul(
            base_point, _upper_left_block(matrix_mn_e, p)
        ) + Matrices.mul(matrix_q, _lower_left_block(matrix_mn_e, p))
        return exp

    @staticmethod
    def retraction(tangent_vec, base_point):
        r"""Compute the retraction of a tangent vector.

        This computation is based on the QR-decomposition.

        e.g. :math:`P_x(V) = qf(X + V)`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, p]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.

        Returns
        -------
        exp : array-like, shape=[..., n, p]
            Point in the Stiefel manifold equal to the retraction
            of tangent_vec at the base point.
        """
        matrix_q, matrix_r = gs.linalg.qr(base_point + tangent_vec)

        diagonal = gs.diagonal(matrix_r, axis1=-2, axis2=-1)
        sign = gs.sign(gs.sign(diagonal) + 0.5)
        diag = algebra_utils.from_vector_to_diagonal_matrix(sign)
        result = Matrices.mul(matrix_q, diag)

        return result

    @staticmethod
    def _matrix_r_single(matrix_m):
        def _make_minor(i, matrix):
            return matrix[: i + 1, : i + 1]

        def _make_column_r(i, matrix, columns_list):
            if i == 0:
                return gs.array([1.0 / matrix[0, 0]])

            matrix_m_i = _make_minor(i, matrix_m)
            inv_matrix_m_i = gs.linalg.inv(matrix_m_i)
            b_i = _make_b(i, matrix_m, columns_list)
            column_r_i = gs.matvec(inv_matrix_m_i, b_i)

            if column_r_i[i] <= 0:
                raise ValueError("(r_i)_i <= 0")
            return column_r_i

        def _make_b(i, matrix, columns_list):
            return gs.array(
                [-gs.dot(matrix[i, : j + 1], columns_list[j]) for j in range(i)] + [1.0]
            )

        n = matrix_m.shape[-1]

        columns_list = []
        matrix_r = gs.zeros((n, n))
        for j in range(n):
            column_r_j = _make_column_r(j, matrix_m, columns_list)
            columns_list.append(column_r_j)

            matrix_r[: len(column_r_j), j] = column_r_j

        return matrix_r

    def lifting(self, point, base_point):
        r"""Compute the lifting of a point.

        This computation is based on the QR-decomposion.

        e.g. :math:`P_x^{-1}(Q) = QR - X`.

        Parameters
        ----------
        point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.
        base_point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.

        Returns
        -------
        log : array-like, shape=[..., dim + 1]
            Tangent vector at the base point equal to the lifting
            of point at the base point.
        """
        point, base_point = gs.broadcast_arrays(point, base_point)

        matrix_m = gs.matmul(gs.transpose(base_point), point)

        if gs.any(matrix_m[..., 0, 0] < 0.0):
            raise ValueError("Algorithm does no work if m11 <= 0.")

        if point.ndim == 2:
            matrix_r = self._matrix_r_single(matrix_m)
        else:
            matrix_r = gs.stack([self._matrix_r_single(matrix) for matrix in matrix_m])

        return gs.matmul(point, matrix_r) - base_point

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image.
        In this case the exact injectivity radius is not known, and we use here
        a lower bound given by [Rentmeesters2015]_.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, p]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.

        References
        ----------
        .. [Rentmeesters2015] Rentmeesters, Quentin. “Algorithms for Data
            Fitting on Some Common Homogeneous Spaces.” UCL - Université
            Catholique de Louvain, 2013.
            https://dial.uclouvain.be/pr/boreal/object/boreal:132587.
        """
        radius = gs.array(0.89 * gs.pi)
        return repeat_out(self._space.point_ndim, radius, base_point)


class _ZimmermannLogSolver(LogSolver):
    """Compute the Stiefel canonical logarithm using Zimmermann's algorithm.

    Implement the algorithm described in [ZR2017]_. Construct an orthogonal
    completion from the base and target frames, then iteratively adjust its
    free columns until the lower-right block of its matrix logarithm is
    approximately zero.

    Parameters
    ----------
    space : Stiefel
        Stiefel manifold equipped with the canonical metric.
    max_iter : int
        Maximum number of iterations for each pair of frames.
        Default: 500.
    tol : float
        Convergence tolerance for the Frobenius norm of the lower-right
        block of the matrix logarithm.
        Default: 1e-8.
    imag_tol : float
        Threshold for the maximum absolute imaginary component of the
        computed matrix logarithm. Discard imaginary components below
        this threshold; otherwise, raise a ValueError.
        Default: 1e-6.

    References
    ----------
    .. [ZR2017] Zimmermann, Ralf. "A Matrix-Algebraic Algorithm for the
        Riemannian Logarithm on the Stiefel Manifold under the Canonical
        Metric." SIAM Journal on Matrix Analysis and Applications,
        38(2), 322–342, 2017.
        https://doi.org/10.1137/16M1074485
    """

    def __init__(self, space, max_iter=500, tol=1e-8, imag_tol=1e-6):
        super().__init__()
        self._space = space

        self.max_iter = max_iter
        self.tol = tol
        self.imag_tol = imag_tol

    @staticmethod
    def _normal_component_qr(point, base_point, matrix_m):
        """Compute the QR decomposition of the normal component of a point.

        Parameters
        ----------
        point : array-like, shape=[..., n, p]
        base_point : array-like, shape=[..., n, p]
        matrix_m : array-like

        Returns
        -------
        matrix_q : array-like
        matrix_n : array-like
        """
        matrix_k = point - gs.matmul(base_point, matrix_m)
        matrix_q, matrix_n = gs.linalg.qr(matrix_k)
        return matrix_q, matrix_n

    def _initial_completion(self, matrix_m, matrix_n):
        """Construct an initial special-orthogonal completion.

        Parameters
        ----------
        matrix_m : array-like, shape=[..., p, p]
            Upper block of the prescribed first p columns.
        matrix_n : array-like, shape=[..., p, p]
            Lower block of the prescribed first p columns.

        Returns
        -------
        matrix_v : array-like, shape=[..., 2 * p, 2 * p]
            Orthogonal completion with positive determinant and first p
            columns equal to the vertical concatenation of the input blocks.
        """
        matrix_v = self._orthogonal_completion(matrix_m, matrix_n)
        return self._align_orthogonal_completion(matrix_v, matrix_m, matrix_n)

    @staticmethod
    def _orthogonal_completion(matrix_m, matrix_n):
        """Construct an orthogonal completion using complete QR decomposition.

        Parameters
        ----------
        matrix_m : array-like, shape=[..., p, p]
            Upper block of a matrix with orthonormal columns.
        matrix_n : array-like, shape=[..., p, p]
            Lower block of a matrix with orthonormal columns.

        Returns
        -------
        matrix_v : array-like, shape=[..., 2 * p, 2 * p]
            Orthogonal matrix whose first p columns span the column space
            of the vertically concatenated input blocks.

        Notes
        -----
        QR decomposition may change the signs of the prescribed columns.
        """
        matrix_w = _concatenate_rows(matrix_m, matrix_n)
        matrix_v, _ = gs.linalg.qr(matrix_w, mode="complete")

        return matrix_v

    @staticmethod
    def _align_orthogonal_completion(matrix_v, matrix_m, matrix_n):
        """Align the free columns of an orthogonal completion using Procrustes.

        Minimize the Frobenius distance of the lower-right block to identity,
        subject to the completed matrix having positive determinant. Preserve
        the first p columns, specified by matrix_m and matrix_n.

        Parameters
        ----------
        matrix_v : array-like, shape=[..., 2 * p, 2 * p]
            Initial orthogonal completion.
        matrix_m : array-like, shape=[..., p, p]
            Upper block of the prescribed first p columns.
        matrix_n : array-like, shape=[..., p, p]
            Lower block of the prescribed first p columns.

        Returns
        -------
        matrix_v : array-like, shape=[..., 2 * p, 2 * p]
            Orthogonal completion with positive determinant and aligned
            lower-right block.
        """
        p = matrix_m.shape[-1]

        matrix_w = _concatenate_rows(matrix_m, matrix_n)
        matrix_z = _lower_right_block(matrix_v, p)
        matrix_d, _, matrix_rt = gs.linalg.svd(matrix_z)
        matrix_r = gs.transpose(matrix_rt)

        matrix_rd = Matrices.mul(matrix_r, gs.transpose(matrix_d))

        sub_matrix_v = gs.matmul(_right_block(matrix_v, p), matrix_rd)
        matrix_v_final = _concatenate_columns(matrix_w, sub_matrix_v)

        det = gs.linalg.det(matrix_v_final)
        sign = gs.concatenate(
            [
                gs.ones(det.shape + (p - 1,), dtype=matrix_v.dtype),
                gs.where(det < 0, -1.0, 1.0)[..., None],
            ],
            axis=-1,
        )

        matrix_rd = Matrices.mul(
            matrix_r * sign[..., None, :],
            gs.transpose(matrix_d),
        )
        sub_matrix_v = gs.matmul(_right_block(matrix_v, p), matrix_rd)
        return _concatenate_columns(matrix_w, sub_matrix_v)

    def _iter_log(self, matrix_v):
        """Compute matrix logarithms for arbitrary batch shapes."""
        if matrix_v.ndim == 2:
            return self._iter_log_single(matrix_v)

        matrix_shape = matrix_v.shape[-2:]
        flat_matrix_v = gs.reshape(matrix_v, (-1,) + matrix_shape)
        matrix_lv = gs.stack(
            [self._iter_log_single(matrix) for matrix in flat_matrix_v]
        )
        return gs.reshape(matrix_lv, matrix_v.shape)

    def _iter_log_single(self, matrix_v):
        """Eliminate the lower-right block of the matrix logarithm iteratively.

        Update the last p columns of the orthogonal completion while preserving
        the first p columns. Stop when the Frobenius norm of the lower-right
        block of its logarithm is at most self.tol.

        Parameters
        ----------
        matrix_v : array-like, shape=[2 * p, 2 * p]
            Initial orthogonal completion with positive determinant.

        Returns
        -------
        matrix_lv : array-like, shape=[2 * p, 2 * p]
            Matrix logarithm of the completion at the last evaluated iterate.
            Its lower-right block is approximately zero if convergence succeeds.

        Raises
        ------
        ValueError
            If the maximum absolute imaginary component of the returned
            matrix logarithm is at least self.imag_tol.

        Warns
        -----
        UserWarning
            If convergence is not reached within self.max_iter iterations.
        """
        p = self._space.p
        matrix_lv = gs.zeros_like(matrix_v)

        for _ in range(self.max_iter):
            matrix_lv = gs.linalg.logm(matrix_v)

            matrix_c = _lower_right_block(matrix_lv, p)
            norm_matrix_c = gs.linalg.norm(matrix_c)
            if norm_matrix_c <= self.tol:
                break

            matrix_phi = gs.linalg.expm(-Matrices.to_skew_symmetric(matrix_c))
            aux_matrix = gs.matmul(_right_block(matrix_v, p), matrix_phi)
            matrix_v = _concatenate_columns(_left_block(matrix_v, p), aux_matrix)

        else:
            warnings.warn("`log` hasn't converged.")

        if gs.is_complex(matrix_lv):
            max_imag = gs.amax(gs.abs(gs.imag(matrix_lv)))
            if max_imag < self.imag_tol:
                matrix_lv = gs.real(matrix_lv)
            else:
                raise ValueError(f"Non-neglible imaginary part. max is {max_imag}")

        return matrix_lv

    def log(self, point, base_point):
        """Compute the Riemannian logarithm of a point.

        When p=n, the space St(n,n)~O(n) has two non connected sheets: the
        log is only defined for data from the same sheet.
        For p<n, the space St(n,p)~O(n)/O(n-p)~SO(n)/SO(n-p) is connected.
        Based on [ZR2017]_.

        Parameters
        ----------
        point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.
        base_point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.

        Returns
        -------
        log : array-like, shape=[..., n, p]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.

        References
        ----------
        .. [ZR2017] Zimmermann, Ralf. "A Matrix-Algebraic Algorithm for the
            Riemannian Logarithm on the Stiefel Manifold under the Canonical
            Metric" SIAM J. Matrix Anal. & Appl., 38(2), 322–342, 2017.
            https://arxiv.org/pdf/1604.05054.pdf
        """
        n, p = self._space.n, self._space.p
        if p == n:
            if gs.any(gs.linalg.det(point) * gs.linalg.det(base_point) <= 0.0):
                raise ValueError("Points from different sheets in log")

        matrix_m = gs.matmul(gs.transpose(base_point), point)

        if gs.any(gs.all(gs.isclose(matrix_m, -gs.eye(p)), axis=(-2, -1))):
            raise ValueError(
                "The Logarithm is not well-defined for antipodal points: "
                f"{point} and {base_point}."
            )

        matrix_q, matrix_n = self._normal_component_qr(point, base_point, matrix_m)

        matrix_v = self._initial_completion(matrix_m, matrix_n)
        matrix_lv = self._iter_log(matrix_v)

        matrix_xv = gs.matmul(base_point, _upper_left_block(matrix_lv, p))
        matrix_qv = gs.matmul(matrix_q, _lower_left_block(matrix_lv, p))

        return matrix_xv + matrix_qv
