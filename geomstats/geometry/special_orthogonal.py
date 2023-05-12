"""Exposes the `SpecialOrthogonal` group class.

Lead authors: Nicolas Guigui and Nina Miolane.
"""

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats.geometry.base import LevelSet
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.invariant_metric import BiInvariantMetric
from geomstats.geometry.lie_group import LieGroup, MatrixLieGroup
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices

ATOL = 1e-5

TAYLOR_COEFFS_1_AT_PI = [
    0.0,
    -gs.pi / 4.0,
    -1.0 / 4.0,
    -gs.pi / 48.0,
    -1.0 / 48.0,
    -gs.pi / 480.0,
    -1.0 / 480.0,
]


class _SpecialOrthogonalMatrices(MatrixLieGroup, LevelSet):
    """Class for special orthogonal groups in matrix representation.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n, equip=True):
        self.n = n
        self._value = gs.eye(n)

        super().__init__(
            dim=int((n * (n - 1)) / 2),
            representation_dim=n,
            lie_algebra=SkewSymmetricMatrices(n=n),
            default_coords_type="extrinsic",
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return BiInvariantMetric

    def _define_embedding_space(self):
        return GeneralLinear(self.n, positive_det=True)

    def _aux_submersion(self, point):
        return Matrices.mul(Matrices.transpose(point), point)

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]

        Returns
        -------
        submersed_point : array-like, shape=[..., n, n]
        """
        return self._aux_submersion(point) - self._value

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
        point : array-like, shape=[..., n, n]

        Returns
        -------
        submersed_vector : array-like, shape=[..., n, n]
        """
        return 2 * Matrices.to_symmetric(
            Matrices.mul(Matrices.transpose(point), vector)
        )

    @classmethod
    def inverse(cls, point):
        """Return the transpose matrix of point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point in SO(n).

        Returns
        -------
        inverse : array-like, shape=[..., n, n]
            Inverse.
        """
        return Matrices.transpose(point)

    def projection(self, point):
        """Project a matrix on SO(n) by minimizing the Frobenius norm.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        rot_mat : array-like, shape=[..., n, n]
            Rotation matrix.
        """
        aux_mat = self._aux_submersion(point)
        inv_sqrt_mat = SymmetricMatrices.powerm(aux_mat, -1 / 2)
        rotation_mat = Matrices.mul(point, inv_sqrt_mat)
        det = gs.linalg.det(rotation_mat)
        return utils.flip_determinant(rotation_mat, det)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in SO(n) using a normal distribution (not the Haar measure).

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Unused.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled on the SO(n).
        """
        return self.random_uniform(n_samples)

    def random_uniform(self, n_samples=1):
        """Sample in SO(n) using a normal distribution (not the Haar measure).

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        tol : unused

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled on the SO(n).
        """
        if n_samples == 1:
            size = (self.n, self.n)
        else:
            size = (n_samples, self.n, self.n)
        random_mat = gs.random.normal(size=size)
        rotation_mat, _ = gs.linalg.qr(random_mat)
        det = gs.linalg.det(rotation_mat)
        return utils.flip_determinant(rotation_mat, det)

    def skew_matrix_from_vector(self, vec):
        """Get the skew-symmetric matrix derived from the vector.

        In nD, fill a skew-symmetric matrix with the values of the vector.

        Parameters
        ----------
        vec : array-like, shape=[..., dim]
            Vector.

        Returns
        -------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.
        """
        return self.lie_algebra.matrix_representation(vec)

    def vector_from_skew_matrix(self, skew_mat):
        """Derive a vector from the skew-symmetric matrix.

        In 3D, compute the vector defining the cross product
        associated to the skew-symmetric matrix skew mat.

        Parameters
        ----------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.

        Returns
        -------
        vec : array-like, shape=[..., dim]
            Vector.
        """
        return self.lie_algebra.basis_representation(skew_mat)

    def rotation_vector_from_matrix(self, rot_mat):
        r"""Convert rotation matrix (in 2D or 3D) to rotation vector.

        Get the angle through the atan2 function:

        Parameters
        ----------
        rot_mat : array-like, shape=[..., 2, 2]
            Rotation matrix.

        Returns
        -------
        regularized_rot_vec : array-like, shape=[..., 1]
            Rotation vector.
        """
        if self.n not in (2, 3):
            raise NotImplementedError(
                "The function matrix_from_rotation_vector is not "
                "implemented if n is not in 2 or 3."
            )
        so_vector = (
            _SpecialOrthogonal2Vectors()
            if self.n == 2
            else _SpecialOrthogonal3Vectors()
        )
        return so_vector.rotation_vector_from_matrix(rot_mat)

    def matrix_from_rotation_vector(self, rot_vec):
        """Convert rotation vector (2D or 3D) to rotation matrix.

        Parameters
        ----------
        rot_vec: array-like, shape=[..., 1]
            Rotation vector.

        Returns
        -------
        rot_mat: array-like, shape=[..., 2, 2]
            Rotation matrix.
        """
        if self.n not in (2, 3):
            raise NotImplementedError(
                "The function matrix_from_rotation_vector is not "
                "implemented if n is not in 2 or 3."
            )
        so_vector = (
            _SpecialOrthogonal2Vectors()
            if self.n == 2
            else _SpecialOrthogonal3Vectors()
        )
        return so_vector.matrix_from_rotation_vector(rot_vec)

    @staticmethod
    def are_antipodals(rotation_mat1, rotation_mat2):
        """Determine if two rotation matrices are antipodals.

        Parameters
        ----------
        rotation_mat1 : array-like, shape=[..., n, n]
            Rotation matrix.
        rotation_mat2 : array-like, shape=[..., n, n]
            Rotation matrix.

        Returns
        -------
        _ : array-like, shape=[...,]
            Boolean determining if each pair of rotation
            matrices corresponds to a pair of antipodal rotation
            matrices.
        """
        sq_rot_mat1 = gs.matmul(rotation_mat1, rotation_mat1)
        sq_rot_mat2 = gs.matmul(rotation_mat2, rotation_mat2)
        are_different = ~gs.all(gs.isclose(rotation_mat1, rotation_mat2), axis=(-2, -1))

        return are_different & gs.all(
            gs.isclose(sq_rot_mat1, sq_rot_mat2), axis=(-2, -1)
        )

    def log(self, point, base_point=None):
        r"""
        Compute the group logarithm of point at base_point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Rotation matrix.
        base_point : array-like, shape=[..., n, n]
            Rotation matrix.
            Optional, defaults to identity if None.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Matrix such that `exp(tangent_vec, base_point) = point`.

        Notes
        -----
        Denoting `point` by :math:`g` and `base_point` by :math:`h`,
        the output satisfies:

        .. math::

            g = \exp(\log(g, h), h)
        """
        if base_point is None:
            base_point = self.identity
        if gs.any(self.are_antipodals(point, base_point)):
            raise ValueError(
                "The Group Logarithm is not well-defined for"
                f" antipodal rotation matrices: {point} and"
                f"{base_point}."
            )
        return super().log(point, base_point)


class _SpecialOrthogonalVectors(LieGroup):
    r"""Class for the special orthogonal groups SO({2,3}) in vector form.

    i.e. the Lie groups of planar and 3D rotations. This class is specific to
    the vector representation of rotations. For the matrix representation use
    the SpecialOrthogonal class and set `n=2` or `n=3`.

    This class represents the Lie group :math:`SO(2)` or :math:`SO(3)`, whose
    Lie algebra is the space of skew symmetric matrices in 2D and 3D respectively.

    This class uses the vector representation to represent points in
    :math:`SO(2)` or :math:`SO(3)`, i.e. a point on the Lie group is represented
    by a vector of size `dim`. Note that the vector actually corresponds to
    the group logarithm of the point at the identity. Hence, in this case, the
    Lie algebra of the Lie group is also equal to the class.

    Parameters
    ----------
    epsilon : float
        Precision to use for calculations involving potential divison by 0 in
        rotations.
        Optional, default: 0.
    """

    def __init__(self, n, epsilon=0.0, equip=True):
        dim = n * (n - 1) // 2
        super().__init__(dim=dim, shape=(dim,), lie_algebra=self, equip=equip)

        self.n = n
        self.epsilon = epsilon

        self._skew_sym_mat = SkewSymmetricMatrices(self.n)

    @property
    def identity(self):
        """Identity of the group.

        Returns
        -------
        identity : array-like, shape=[1,]
            Identity.
        """
        return gs.zeros(self.dim)

    def belongs(self, point, atol=ATOL):
        """Evaluate if a point belongs to SO(2) or SO(3).

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to check.
        atol : unused

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean indicating whether point belongs to SO(3).
        """
        belongs = self.shape == point.shape[-self.point_ndim :]
        shape = point.shape[: -self.point_ndim]
        if belongs:
            return gs.ones(shape, dtype=bool)
        return gs.zeros(shape, dtype=bool)

    def projection(self, point):
        """Project a matrix on SO(2) or SO(3) using the Frobenius norm.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        rot_mat : array-like, shape=[..., n, n]
            Rotation matrix.
        """
        mat = point

        mat_unitary_u, _, mat_unitary_v = gs.linalg.svd(mat)
        rot_mat = Matrices.mul(mat_unitary_u, mat_unitary_v)
        mask = gs.less(gs.linalg.det(rot_mat), 0.0)
        mask_float = gs.cast(mask, mat.dtype) + self.epsilon

        diag = gs.concatenate((gs.ones(self.n - 1), -gs.ones(1)), axis=0)
        diag = utils.from_vector_to_diagonal_matrix(diag) + self.epsilon

        aux_mat = Matrices.mul(mat_unitary_u, diag)
        rot_mat = rot_mat + gs.einsum(
            "...,...jk->...jk", mask_float, Matrices.mul(aux_mat, mat_unitary_v)
        )
        return rot_mat

    def inverse(self, point):
        """Compute the group inverse in SO(2) or SO(3).

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        inv_point : array-like, shape=[..., dim]
            Inverse.
        """
        return -self.regularize(point)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in SO(n) using a uniform distribution (not the Haar measure).

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Unused.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled on the SO(n).
        """
        return gs.squeeze(gs.random.rand(n_samples, 3))

    def exp_from_identity(self, tangent_vec):
        """Compute the group exponential of the tangent vector at the identity.

        As rotations are represented by their rotation vector,
        which corresponds to the element `X` in the Lie Algebra such that
        `exp(X) = R`, this methods returns its input without change.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.

        Returns
        -------
        point : array-like, shape=[..., dim]
            Point.
        """
        return self.regularize(tangent_vec)

    def log_from_identity(self, point):
        """Compute the group logarithm of the point at the identity.

        As rotations are represented by their rotation vector,
        which corresponds to the element `X` in the Lie Algebra such that
        `exp(X) = R`, this methods returns its input after regularization.


        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Group logarithm.
        """
        return self.regularize(point)

    def skew_matrix_from_vector(self, vec):
        """Get the skew-symmetric matrix derived from the vector.

        In 3D, compute the skew-symmetric matrix, known as the cross-product of
        a vector, associated to the vector `vec`.

        Parameters
        ----------
        vec : array-like, shape=[..., dim]
            Vector.

        Returns
        -------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.
        """
        return self._skew_sym_mat.matrix_representation(vec)

    def vector_from_skew_matrix(self, skew_mat):
        """Derive a vector from the skew-symmetric matrix.

        In 3D, compute the vector defining the cross product
        associated to the skew-symmetric matrix skew mat.

        Parameters
        ----------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.

        Returns
        -------
        vec : array-like, shape=[..., dim]
            Vector.
        """
        return self._skew_sym_mat.basis_representation(skew_mat)

    def to_tangent(self, vector, base_point=None):
        """Project a vector onto the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector to project.
        base_point : array-like, shape=[..., dim]
            Point of the group.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        tangent_vec = self.regularize_tangent_vec(vector, base_point)
        if base_point is not None and base_point.ndim > vector.ndim:
            return gs.broadcast_to(tangent_vec, base_point.shape)

        return tangent_vec

    def regularize_tangent_vec_at_identity(self, tangent_vec):
        """Regularize a tangent vector at the identity.

        In 2D, regularize a tangent_vector by getting its norm at the identity,
        to be less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.

        Returns
        -------
        regularized_vec : array-like, shape=[..., 1]
            Regularized tangent vector.
        """
        return self.regularize(tangent_vec)

    def regularize_tangent_vec(self, tangent_vec, base_point):
        """Regularize tangent vector at a base point.

        In 2D, regularize a tangent_vector by getting the norm of its parallel
        transport to the identity, determined by the metric, less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[...,1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Point on the manifold.

        Returns
        -------
        regularized_tangent_vec : array-like, shape=[..., 1]
            Regularized tangent vector.
        """
        return self.regularize_tangent_vec_at_identity(tangent_vec)


class _SpecialOrthogonal2Vectors(_SpecialOrthogonalVectors):
    """Class for the special orthogonal group SO(2) in vector representation.

    i.e. the Lie group of planar rotations. This class is specific to the
    vector representation of rotations. For the matrix representation use the
    SpecialOrthogonal class and set `n=2`.

    Parameters
    ----------
    epsilon : float
        Precision to use for calculations involving potential divison by 0 in
        rotations.
        Optional, default: 0.
    """

    def __init__(self, epsilon=0.0, equip=True):
        super().__init__(
            n=2,
            epsilon=epsilon,
            equip=equip,
        )

    def regularize(self, point):
        """Regularize a point to be in accordance with convention.

        In 2D, regularize the norm of the rotation angle,
        to be between -pi and pi, following the axis-angle
        representation's convention.

        If the angle angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        Parameters
        ----------
        point : array-like, shape=[...,1]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., 1]
            Regularized point.
        """
        regularized_point = point
        regularized_point = gs.mod(regularized_point, 2 * gs.pi)
        regularized_point = gs.where(
            regularized_point < gs.pi, regularized_point, regularized_point - 2 * gs.pi
        )
        return regularized_point

    def rotation_vector_from_matrix(self, rot_mat):
        r"""Convert rotation matrix (in 2D) to rotation vector (axis-angle).

        Get the angle through the atan2 function:

        Parameters
        ----------
        rot_mat : array-like, shape=[..., 2, 2]
            Rotation matrix.

        Returns
        -------
        regularized_rot_vec : array-like, shape=[..., 1]
            Rotation vector.
        """
        rot_vec = gs.arctan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])
        return self.regularize(rot_vec[..., None])

    def matrix_from_rotation_vector(self, rot_vec):
        """Convert rotation vector to rotation matrix.

        Parameters
        ----------
        rot_vec: array-like, shape=[..., 1]
            Rotation vector.

        Returns
        -------
        rot_mat: array-like, shape=[..., 2, 2]
            Rotation matrix.
        """
        rot_vec = self.regularize(rot_vec)

        cos_term = gs.cos(rot_vec)
        cos_matrix = gs.einsum("...l,ij->...ij", cos_term, gs.eye(2))
        sin_term = gs.sin(rot_vec)
        sin_matrix = self.skew_matrix_from_vector(sin_term)
        return cos_matrix + sin_matrix

    def compose(self, point_a, point_b):
        """Compose two elements of SO(3).

        Parameters
        ----------
        point_a : array-like, shape=[..., 3]
        point_b : array-like, shape=[..., 3]

        Returns
        -------
        point_prod : array-like, shape=[..., 3]
        """
        point_a = self.regularize(point_a)
        point_b = self.regularize(point_b)

        point_prod = point_a + point_b
        point_prod = self.regularize(point_prod)

        return point_prod

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in SO(2) using the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Unused.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled on the SO(2).
        """
        return self.random_uniform(n_samples)

    def random_uniform(self, n_samples=1):
        """Sample in SO(2) with the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        point : array-like, shape=[..., 1]
            Sample.
        """
        random_point = (gs.random.rand(n_samples, self.dim) * 2 - 1) * gs.pi
        random_point = self.regularize(random_point)

        if n_samples == 1:
            random_point = gs.squeeze(random_point, axis=0)

        return random_point

    def exp(self, tangent_vec, base_point=None):
        """Compute the group exponential.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Point from which the exponential is computed.

        Returns
        -------
        point : array-like, shape=[..., 1]
            Group exponential.
        """
        return self.regularize(tangent_vec + base_point)

    def log(self, point, base_point=None):
        """Compute the group logarithm.

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point.
        base_point : array-like, shape=[..., 1]
            Point from which the log is computed.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Group logarithm.
        """
        return self.regularize(point - base_point)


class _SpecialOrthogonal3Vectors(_SpecialOrthogonalVectors):
    """Class for the special orthogonal group SO(3) in vector representation.

    i.e. the Lie group of rotations. This class is specific to the vector
    representation of rotations. For the matrix representation use the
    SpecialOrthogonal class and set `n=3`.

    Parameters
    ----------
    epsilon : float
        Precision to use for calculations involving potential divison by 0 in
        rotations.
        Optional, default: 0.
    """

    def __init__(self, epsilon=0.0, equip=True):
        super().__init__(n=3, epsilon=epsilon, equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return BiInvariantMetric

    def regularize(self, point):
        """Regularize a point to be in accordance with convention.

        In 3D, regularize the norm of the rotation vector,
        to be between 0 and pi, following the axis-angle
        representation's convention.

        If the angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        Parameters
        ----------
        point : array-like, shape=[...,3]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., 3]
            Regularized point.
        """
        theta = gs.linalg.norm(point, axis=-1)
        k = gs.floor(theta / 2.0 / gs.pi)

        # angle in [0;2pi)
        angle = theta - 2 * k * gs.pi

        # this avoids dividing by 0
        theta_eps = gs.where(gs.isclose(theta, 0.0), 1.0, theta)

        # angle in [0, pi]
        normalized_angle = gs.where(angle <= gs.pi, angle, 2 * gs.pi - angle)
        norm_ratio = gs.where(gs.isclose(theta, 0.0), 1.0, normalized_angle / theta_eps)

        # reverse sign if angle was greater than pi
        norm_ratio = gs.where(angle > gs.pi, -norm_ratio, norm_ratio)
        return gs.einsum("...,...i->...i", norm_ratio, point)

    def regularize_tangent_vec_at_identity(self, tangent_vec):
        """Regularize a tangent vector at the identity.

        In 3D, regularize a tangent_vector by getting its norm at the identity,
        determined by the metric, to be less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 3]
            Tangent vector at base point.

        Returns
        -------
        regularized_vec : array-like, shape=[..., 3]
            Regularized tangent vector.
        """
        if not hasattr(self, "metric"):
            return self.regularize(tangent_vec)

        tangent_vec_metric_norm = self.metric.norm(tangent_vec)
        tangent_vec_canonical_norm = gs.linalg.norm(tangent_vec, axis=-1)

        # This avoids dividing by 0
        norm_eps = gs.where(
            tangent_vec_canonical_norm == 0, gs.atol, tangent_vec_canonical_norm
        )
        coef = gs.where(
            tangent_vec_canonical_norm == 0.0, 1.0, tangent_vec_metric_norm / norm_eps
        )
        coef_tangent_vec = gs.einsum("...,...i->...i", coef, tangent_vec)

        regularized_vec = self.regularize(coef_tangent_vec)
        return gs.einsum("...,...i->...i", 1.0 / coef, regularized_vec)

    def regularize_tangent_vec(self, tangent_vec, base_point):
        """Regularize tangent vector at a base point.

        In 3D, regularize a tangent_vector by getting the norm of its parallel
        transport to the identity, determined by the metric, less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[...,3]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 3]
            Point on the manifold.

        Returns
        -------
        regularized_tangent_vec : array-like, shape=[..., 3]
            Regularized tangent vector.
        """
        base_point = self.regularize(base_point)

        tangent_vec_at_id = self.tangent_translation_map(
            base_point, left=self.metric.left, inverse=True
        )(tangent_vec)

        tangent_vec_at_id = self.regularize_tangent_vec_at_identity(tangent_vec_at_id)

        regularized_tangent_vec = self.tangent_translation_map(
            base_point, left=self.metric.left
        )(tangent_vec_at_id)

        return regularized_tangent_vec

    def rotation_vector_from_matrix(self, rot_mat):
        r"""Convert rotation matrix (in 3D) to rotation vector (axis-angle).

        Get the angle through the trace of the rotation matrix:
        The eigenvalues are:
        :math:`\{1, \cos(angle) + i \sin(angle), \cos(angle) - i \sin(angle)\}`
        so that:
        :math:`trace = 1 + 2 \cos(angle), \{-1 \leq trace \leq 3\}`

        The rotation vector is the vector associated to the skew-symmetric
        matrix
        :math:`S_r = \frac{angle}{(2 * \sin(angle) ) (R - R^T)}`

        For the edge case where the angle is close to pi,
        the rotation vector (up to sign) is derived by using the following
        equality (see the Axis-angle representation on Wikipedia):
        :math:`outer(r, r) = \frac{1}{2} (R + I_3)`
        In nD, the rotation vector stores the :math:`n(n-1)/2` values
        of the skew-symmetric matrix representing the rotation.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., n, n]
            Rotation matrix.

        Returns
        -------
        regularized_rot_vec : array-like, shape=[..., 3]
            Rotation vector.
        """
        is_vec = gs.ndim(rot_mat) > 2

        trace = gs.trace(rot_mat)
        trace_num = gs.clip(trace, -1, 3)
        angle = gs.arccos(0.5 * (trace_num - 1))

        rot_mat_transpose = Matrices.transpose(rot_mat)
        rot_vec_not_pi = self.vector_from_skew_matrix(rot_mat - rot_mat_transpose)

        mask_0 = gs.cast(gs.isclose(angle, 0.0), angle.dtype)
        mask_pi = gs.cast(gs.isclose(angle, gs.pi, atol=1e-2), angle.dtype)
        mask_else = (1 - mask_0) * (1 - mask_pi)

        numerator = 0.5 * mask_0 + angle * mask_else
        denominator = (
            (1 - angle**2 / 6) * mask_0 + 2 * gs.sin(angle) * mask_else + mask_pi
        )
        rot_vec_not_pi = gs.einsum(
            "...,...i->...i", numerator / denominator, rot_vec_not_pi
        )

        vector_outer = 0.5 * (gs.eye(3) + rot_mat)
        vector_outer = gs.set_diag(
            vector_outer, gs.maximum(0.0, gs.diagonal(vector_outer, axis1=-2, axis2=-1))
        )
        squared_diag_comp = gs.diagonal(vector_outer, axis1=-2, axis2=-1)
        diag_comp = gs.sqrt(squared_diag_comp)
        norm_line = gs.linalg.norm(vector_outer, axis=-1)
        max_line_index = gs.argmax(norm_line, axis=-1)

        if is_vec:
            selected_line = gs.get_slice(
                vector_outer, (range(rot_mat.shape[0]), max_line_index)
            )
        else:
            selected_line = vector_outer[..., max_line_index]
        signs = gs.sign(selected_line)
        rot_vec_pi = gs.einsum("...,...i,...i->...i", angle, signs, diag_comp)

        rot_vec = rot_vec_not_pi + gs.einsum("...,...i->...i", mask_pi, rot_vec_pi)

        return self.regularize(rot_vec)

    def matrix_from_rotation_vector(self, rot_vec):
        """Convert rotation vector to rotation matrix.

        Parameters
        ----------
        rot_vec: array-like, shape=[..., 3]
            Rotation vector.

        Returns
        -------
        rot_mat: array-like, shape=[..., 3]
            Rotation matrix.
        """
        rot_vec = self.regularize(rot_vec)

        squared_angle = gs.sum(rot_vec**2, axis=-1)
        skew_rot_vec = self.skew_matrix_from_vector(rot_vec)

        coef_1 = utils.taylor_exp_even_func(squared_angle, utils.sinc_close_0)
        coef_2 = utils.taylor_exp_even_func(squared_angle, utils.cosc_close_0)

        term_1 = gs.eye(self.dim) + gs.einsum("...,...jk->...jk", coef_1, skew_rot_vec)

        squared_skew_rot_vec = Matrices.mul(skew_rot_vec, skew_rot_vec)

        term_2 = gs.einsum("...,...jk->...jk", coef_2, squared_skew_rot_vec)

        return term_1 + term_2

    def quaternion_from_matrix(self, rot_mat):
        """Convert a rotation matrix into a unit quaternion.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., 3, 3]
            Rotation matrix.

        Returns
        -------
        quaternion : array-like, shape=[..., 4]
            Quaternion.
        """
        rot_vec = self.rotation_vector_from_matrix(rot_mat)
        return self.quaternion_from_rotation_vector(rot_vec)

    def quaternion_from_rotation_vector(self, rot_vec):
        """Convert a rotation vector into a unit quaternion.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]
            Rotation vector.

        Returns
        -------
        quaternion : array-like, shape=[..., 4]
            Quaternion.
        """
        rot_vec = self.regularize(rot_vec)

        squared_angle = gs.sum(rot_vec**2, axis=-1)

        coef_cos = utils.taylor_exp_even_func(squared_angle / 4, utils.cos_close_0)
        coef_sinc = 0.5 * utils.taylor_exp_even_func(
            squared_angle / 4, utils.sinc_close_0
        )

        quaternion = gs.concatenate(
            (coef_cos[..., None], gs.einsum("...,...i->...i", coef_sinc, rot_vec)),
            axis=-1,
        )

        return quaternion

    def rotation_vector_from_quaternion(self, quaternion):
        """Convert a unit quaternion into a rotation vector.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]
            Quaternion.

        Returns
        -------
        rot_vec : array-like, shape=[..., 3]
            Rotation vector.
        """
        cos_half_angle = quaternion[..., 0]
        cos_half_angle = gs.clip(cos_half_angle, -1, 1)
        half_angle = gs.arccos(cos_half_angle)

        coef_isinc = 2 * utils.taylor_exp_even_func(
            half_angle**2, utils.inv_sinc_close_0
        )

        rot_vec = gs.einsum("...,...i->...i", coef_isinc, quaternion[..., 1:])

        return self.regularize(rot_vec)

    def matrix_from_quaternion(self, quaternion):
        """Convert a unit quaternion into a rotation vector.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]
            Quaternion.

        Returns
        -------
        rot_mat : array-like, shape=[..., 3]
            Rotation matrix.
        """
        is_vec = quaternion.ndim > 1

        w, x, y, z = gs.hsplit(quaternion, 4)

        column_1 = gs.array(
            [
                w**2 + x**2 - y**2 - z**2,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
            ]
        )

        column_2 = gs.array(
            [
                2 * x * y + 2 * w * z,
                w**2 - x**2 + y**2 - z**2,
                2 * y * z - 2 * w * x,
            ]
        )

        column_3 = gs.array(
            [
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                w**2 - x**2 - y**2 + z**2,
            ]
        )

        if is_vec:
            column_1 = gs.moveaxis(column_1, 0, 1)
            column_2 = gs.moveaxis(column_2, 0, 1)
            column_3 = gs.moveaxis(column_3, 0, 1)

            rot_mat = gs.stack(
                [
                    gs.transpose(gs.hstack(columns))
                    for columns in zip(column_1, column_2, column_3)
                ]
            )

        else:
            rot_mat = gs.transpose(gs.hstack([column_1, column_2, column_3]))

        return rot_mat

    @staticmethod
    def _matrix_from_tait_bryan_angles_extrinsic_xyz(tait_bryan_angles):
        """Convert Tait-Bryan angles to rot mat in extrinsic coords (xyz).

        Convert a rotation given in terms of the tait bryan angles,
        [angle_1, angle_2, angle_3] in extrinsic (fixed) coordinate system
        in order xyz, into a rotation matrix.

        rot_mat = Z(angle_1).Y(angle_2).X(angle_3)
        where:

        - Z(angle_1) is a rotation of angle angle_1 around axis z.
        - Y(angle_2) is a rotation of angle angle_2 around axis y.
        - X(angle_3) is a rotation of angle angle_3 around axis x.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]

        Returns
        -------
        rot_mat : array-like, shape=[..., 3, 3]
        """
        is_vec = tait_bryan_angles.ndim > 1

        angle_1 = tait_bryan_angles[..., 0]
        angle_2 = tait_bryan_angles[..., 1]
        angle_3 = tait_bryan_angles[..., 2]

        cos_angle_1 = gs.cos(angle_1)
        sin_angle_1 = gs.sin(angle_1)
        cos_angle_2 = gs.cos(angle_2)
        sin_angle_2 = gs.sin(angle_2)
        cos_angle_3 = gs.cos(angle_3)
        sin_angle_3 = gs.sin(angle_3)

        column_1 = gs.array(
            [
                [cos_angle_1 * cos_angle_2],
                [cos_angle_2 * sin_angle_1],
                [-sin_angle_2],
            ]
        )
        column_2 = gs.array(
            [
                [(cos_angle_1 * sin_angle_2 * sin_angle_3 - cos_angle_3 * sin_angle_1)],
                [(cos_angle_1 * cos_angle_3 + sin_angle_1 * sin_angle_2 * sin_angle_3)],
                [cos_angle_2 * sin_angle_3],
            ]
        )
        column_3 = gs.array(
            [
                [(sin_angle_1 * sin_angle_3 + cos_angle_1 * cos_angle_3 * sin_angle_2)],
                [(cos_angle_3 * sin_angle_1 * sin_angle_2 - cos_angle_1 * sin_angle_3)],
                [cos_angle_2 * cos_angle_3],
            ]
        )

        if is_vec:
            column_1 = gs.moveaxis(column_1, 2, 0)
            column_2 = gs.moveaxis(column_2, 2, 0)
            column_3 = gs.moveaxis(column_3, 2, 0)

            rot_mat = gs.stack(
                [gs.hstack(columns) for columns in zip(column_1, column_2, column_3)]
            )

        else:
            rot_mat = gs.hstack([column_1, column_2, column_3])

        return rot_mat

    @staticmethod
    def _matrix_from_tait_bryan_angles_extrinsic_zyx(tait_bryan_angles):
        """Convert Tait-Bryan angles to rot mat in extrensic coords (zyx).

        Convert a rotation given in terms of the tait bryan angles,
        [angle_1, angle_2, angle_3] in extrinsic (fixed) coordinate system
        in order zyx, into a rotation matrix.

        rot_mat = X(angle_1).Y(angle_2).Z(angle_3)
        where:

        - X(angle_1) is a rotation of angle angle_1 around axis x.
        - Y(angle_2) is a rotation of angle angle_2 around axis y.
        - Z(angle_3) is a rotation of angle angle_3 around axis z.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]

        Returns
        -------
        rot_mat : array-like, shape=[..., n, n]
        """
        is_vec = tait_bryan_angles.ndim > 1

        angle_1 = tait_bryan_angles[..., 0]
        angle_2 = tait_bryan_angles[..., 1]
        angle_3 = tait_bryan_angles[..., 2]

        cos_angle_1 = gs.cos(angle_1)
        sin_angle_1 = gs.sin(angle_1)
        cos_angle_2 = gs.cos(angle_2)
        sin_angle_2 = gs.sin(angle_2)
        cos_angle_3 = gs.cos(angle_3)
        sin_angle_3 = gs.sin(angle_3)

        column_1 = gs.array(
            [
                [cos_angle_2 * cos_angle_3],
                [(cos_angle_1 * sin_angle_3 + cos_angle_3 * sin_angle_1 * sin_angle_2)],
                [(sin_angle_1 * sin_angle_3 - cos_angle_1 * cos_angle_3 * sin_angle_2)],
            ]
        )

        column_2 = gs.array(
            [
                [-cos_angle_2 * sin_angle_3],
                [(cos_angle_1 * cos_angle_3 - sin_angle_1 * sin_angle_2 * sin_angle_3)],
                [(cos_angle_3 * sin_angle_1 + cos_angle_1 * sin_angle_2 * sin_angle_3)],
            ]
        )

        column_3 = gs.array(
            [
                [sin_angle_2],
                [-cos_angle_2 * sin_angle_1],
                [cos_angle_1 * cos_angle_2],
            ]
        )

        if is_vec:
            column_1 = gs.moveaxis(column_1, 2, 0)
            column_2 = gs.moveaxis(column_2, 2, 0)
            column_3 = gs.moveaxis(column_3, 2, 0)

            rot_mat = gs.stack(
                [gs.hstack(columns) for columns in zip(column_1, column_2, column_3)]
            )

        else:
            rot_mat = gs.hstack([column_1, column_2, column_3])

        return rot_mat

    def matrix_from_tait_bryan_angles(
        self,
        tait_bryan_angles,
        extrinsic=True,
        zyx=True,
    ):
        """Convert Tait-Bryan angles to rot mat in extr or intr coords.

        Convert a rotation given in terms of the tait bryan angles,
        [angle_1, angle_2, angle_3] in extrinsic (fixed) or
        intrinsic (moving) coordinate frame into a rotation matrix.

        If the order is zyx, into the rotation matrix rot_mat:
        rot_mat = X(angle_1).Y(angle_2).Z(angle_3)
        where:

        - X(angle_1) is a rotation of angle angle_1 around axis x.
        - Y(angle_2) is a rotation of angle angle_2 around axis y.
        - Z(angle_3) is a rotation of angle angle_3 around axis z.

        Exchanging 'extrinsic' and 'intrinsic' amounts to
        exchanging the order.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]
        extrinsic : bool
            If False, then 'intrinsic'.
        zyx : bool
            If False, then 'xyz'.

        Returns
        -------
        rot_mat : array-like, shape=[..., n, n]
        """
        if extrinsic and zyx:
            return self._matrix_from_tait_bryan_angles_extrinsic_zyx(tait_bryan_angles)
        if not extrinsic and not zyx:
            tait_bryan_angles_reversed = gs.flip(tait_bryan_angles, axis=-1)
            return self._matrix_from_tait_bryan_angles_extrinsic_zyx(
                tait_bryan_angles_reversed
            )

        if extrinsic and not zyx:
            return self._matrix_from_tait_bryan_angles_extrinsic_xyz(tait_bryan_angles)

        # not extrinsic and zyx
        tait_bryan_angles_reversed = gs.flip(tait_bryan_angles, axis=-1)
        return self._matrix_from_tait_bryan_angles_extrinsic_xyz(
            tait_bryan_angles_reversed
        )

    def tait_bryan_angles_from_matrix(self, rot_mat, extrinsic=True, zyx=True):
        """Convert rot_mat into Tait-Bryan angles.

        Convert a rotation matrix rot_mat into the tait bryan angles,
        [angle_1, angle_2, angle_3] in extrinsic (fixed) coordinate frame,
        for the order zyx, i.e.:
        rot_mat = X(angle_1).Y(angle_2).Z(angle_3)
        where:

        - X(angle_1) is a rotation of angle angle_1 around axis x.
        - Y(angle_2) is a rotation of angle angle_2 around axis y.
        - Z(angle_3) is a rotation of angle angle_3 around axis z.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., n, n]
        extrinsic : bool
            If False, then 'intrinsic'.
        zyx : bool
            If False, then 'xyz'.

        Returns
        -------
        tait_bryan_angles : array-like, shape=[..., 3]
        """
        quaternion = self.quaternion_from_matrix(rot_mat)
        return self.tait_bryan_angles_from_quaternion(
            quaternion, extrinsic=extrinsic, zyx=zyx
        )

    def _quaternion_from_tait_bryan_angles_intrinsic_xyz(self, tait_bryan_angles):
        """Convert Tait-Bryan angles to into unit quaternion.

        Convert a rotation given by Tait-Bryan angles in extrinsic
        coordinate systems and order xyz into a unit quaternion.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]

        Returns
        -------
        quaternion : array-like, shape=[..., 4]
        """
        matrix = self.matrix_from_tait_bryan_angles(
            tait_bryan_angles, extrinsic=False, zyx=False
        )
        return self.quaternion_from_matrix(matrix)

    def quaternion_from_tait_bryan_angles(
        self, tait_bryan_angles, extrinsic=True, zyx=True
    ):
        """Convert a rotation given by Tait-Bryan angles into unit quaternion.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]
        extrinsic : bool
            If False, then 'intrinsic'.
        zyx : bool
            If False, then 'xyz'.

        Returns
        -------
        quat : array-like, shape=[..., 4]
        """
        if extrinsic and zyx:
            tait_bryan_angles_reversed = gs.flip(tait_bryan_angles, axis=-1)
            return self._quaternion_from_tait_bryan_angles_intrinsic_xyz(
                tait_bryan_angles_reversed
            )

        if not extrinsic and not zyx:
            return self._quaternion_from_tait_bryan_angles_intrinsic_xyz(
                tait_bryan_angles
            )

        if extrinsic and not zyx:
            rot_mat = self._matrix_from_tait_bryan_angles_extrinsic_xyz(
                tait_bryan_angles
            )
            return self.quaternion_from_matrix(rot_mat)

        # not extrinsic and zyx
        tait_bryan_angles_reversed = gs.flip(tait_bryan_angles, axis=-1)
        rot_mat = self._matrix_from_tait_bryan_angles_extrinsic_xyz(
            tait_bryan_angles_reversed
        )
        return self.quaternion_from_matrix(rot_mat)

    def rotation_vector_from_tait_bryan_angles(
        self,
        tait_bryan_angles,
        extrinsic=True,
        zyx=True,
    ):
        """Convert rotation given by angle_1, angle_2, angle_3 into rot. vec.

        Convert into axis-angle representation.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]
        extrinsic : bool
            If False, then 'intrinsic'.
        zyx : bool
            If False, then 'xyz'.

        Returns
        -------
        rot_vec : array-like, shape=[..., 3]
        """
        quaternion = self.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic=extrinsic,
            zyx=zyx,
        )
        rot_vec = self.rotation_vector_from_quaternion(quaternion)

        return self.regularize(rot_vec)

    @staticmethod
    def _tait_bryan_angles_from_quaternion_intrinsic_zyx(quaternion):
        """Convert quaternion to tait bryan representation of order zyx.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]

        Returns
        -------
        tait_bryan_angles : array-like, shape=[..., 3]
        """
        w, x, y, z = gs.hsplit(quaternion, 4)
        angle_1 = gs.arctan2(y * z + w * x, 1.0 / 2.0 - (x**2 + y**2))
        angle_2 = gs.arcsin(-2.0 * (x * z - w * y))
        angle_3 = gs.arctan2(x * y + w * z, 1.0 / 2.0 - (y**2 + z**2))
        return gs.concatenate([angle_1, angle_2, angle_3], axis=-1)

    @staticmethod
    def _tait_bryan_angles_from_quaternion_intrinsic_xyz(quaternion):
        """Convert quaternion to tait bryan representation of order xyz.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]

        Returns
        -------
        tait_bryan_angles : array-like, shape=[..., 3]
        """
        w, x, y, z = gs.hsplit(quaternion, 4)

        angle_1 = gs.arctan2(2.0 * (-x * y + w * z), w * w + x * x - y * y - z * z)
        angle_2 = gs.arcsin(2 * (x * z + w * y))
        angle_3 = gs.arctan2(2.0 * (-y * z + w * x), w * w + z * z - x * x - y * y)

        return gs.concatenate([angle_1, angle_2, angle_3], axis=-1)

    def tait_bryan_angles_from_quaternion(self, quaternion, extrinsic=True, zyx=True):
        """Convert quaternion to a rotation in form angle_1, angle_2, angle_3.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]
        extrinsic : bool
            If False, then 'intrinsic'.
        zyx : bool
            If False, then 'xyz'.

        Returns
        -------
        tait_bryan : array-like, shape=[..., 3]
        """
        if extrinsic and zyx:
            tait_bryan = self._tait_bryan_angles_from_quaternion_intrinsic_xyz(
                quaternion
            )
            return gs.flip(tait_bryan, axis=-1)

        if not extrinsic and not zyx:
            return self._tait_bryan_angles_from_quaternion_intrinsic_xyz(quaternion)

        if extrinsic and not zyx:
            tait_bryan = self._tait_bryan_angles_from_quaternion_intrinsic_zyx(
                quaternion
            )
            return gs.flip(tait_bryan, axis=-1)

        # not extrinsic and zyx
        return self._tait_bryan_angles_from_quaternion_intrinsic_zyx(quaternion)

    def tait_bryan_angles_from_rotation_vector(
        self,
        rot_vec,
        extrinsic=True,
        zyx=True,
    ):
        """Convert a rotation vector to a rotation given by Tait-Bryan angles.

        Here the rotation vector is in the axis-angle representation.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]
        extrinsic : bool
            If False, then 'intrinsic'.
        zyx : bool
            If False, then 'xyz'.

        Returns
        -------
        tait_bryan_angles : array-like, shape=[..., 3]
        """
        quaternion = self.quaternion_from_rotation_vector(rot_vec)
        return self.tait_bryan_angles_from_quaternion(
            quaternion, extrinsic=extrinsic, zyx=zyx
        )

    def compose(self, point_a, point_b):
        """Compose two elements of SO(3).

        Parameters
        ----------
        point_a : array-like, shape=[..., 3]
        point_b : array-like, shape=[..., 3]

        Returns
        -------
        point_prod : array-like, shape=[..., 3]
        """
        point_a = self.regularize(point_a)
        point_b = self.regularize(point_b)

        point_a = self.matrix_from_rotation_vector(point_a)
        point_b = self.matrix_from_rotation_vector(point_b)
        point_prod = gs.matmul(point_a, point_b)

        point_prod = self.rotation_vector_from_matrix(point_prod)
        point_prod = self.regularize(point_prod)

        return point_prod

    def jacobian_translation(self, point, left=True):
        """Compute the jacobian matrix corresponding to translation.

        Compute the jacobian matrix of the differential
        of the left/right translations from the identity to point in SO(3).

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point.
        left : bool
            Whether to use left or right invariant metric.
            Optional, default: True.

        Returns
        -------
        jacobian : array-like, shape=[..., 3, 3]
            Jacobian.
        """
        point = self.regularize(point)
        squared_angle = gs.sum(point**2, axis=-1)

        angle = gs.sqrt(squared_angle)
        delta_angle = angle - gs.pi
        approx_at_pi = gs.sum(
            gs.array([TAYLOR_COEFFS_1_AT_PI[k] * delta_angle**k for k in range(1, 7)])
        )
        coef_1 = utils.taylor_exp_even_func(squared_angle / 4, utils.inv_tanc_close_0)
        coef_1 = gs.where(-delta_angle < utils.EPSILON, approx_at_pi, coef_1)

        coef_2 = utils.taylor_exp_even_func(
            squared_angle, utils.var_inv_tanc_close_0, order=4
        )
        squared_angle_ = gs.where(
            squared_angle < utils.EPSILON, utils.EPSILON, squared_angle
        )
        coef_2 = gs.where(
            squared_angle < utils.EPSILON, coef_2, (1 - coef_1) / squared_angle_
        )

        outer_ = gs.outer(point, point)
        sign = 1.0 if left else -1.0

        return (
            gs.einsum("...,...ij->...ij", coef_1, gs.eye(self.dim))
            + gs.einsum("...,...ij->...ij", coef_2, outer_)
            + sign * self.skew_matrix_from_vector(point) / 2.0
        )

    def random_uniform(self, n_samples=1):
        """Sample in SO(3) uniform wrt parameters - not Haar measure.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        point : array-like, shape=[..., 3]
            Sample.
        """
        random_point = gs.random.rand(n_samples, self.dim) * 2 - 1
        random_point = self.regularize(random_point)

        if n_samples == 1:
            random_point = gs.squeeze(random_point, axis=0)

        return random_point

    def lie_bracket(self, tangent_vector_a, tangent_vector_b, base_point=None):
        """Compute the lie bracket of two tangent vectors.

        For matrix Lie groups with tangent vectors A,B at the same base point P
        this is given by (translate to identity, compute commutator, go back)
        :math:`[A,B] = A_P^{-1}B - B_P^{-1}A`

        Parameters
        ----------
        tangent_vector_a : shape=[..., n, n]
            Tangent vector at base point.
        tangent_vector_b : shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.
            Optional, default: None.

        Returns
        -------
        bracket : array-like, shape=[..., n, n]
            Lie bracket.
        """
        out = gs.cross(tangent_vector_a, tangent_vector_b)
        if (
            base_point is not None
            and base_point.ndim > tangent_vector_a.ndim
            and base_point.ndim > tangent_vector_b.ndim
        ):
            return gs.broadcast_to(out, base_point.shape)
        return out


class SpecialOrthogonal:
    r"""Class for the special orthogonal groups.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices : n x n.
    point_type : str, {\'vector\', \'matrix\'}
        Representation of the elements of the group.
    epsilon : float, optional
        precision to use for calculations involving potential divison by 0 in
        rotations
        default: 0
    """

    def __new__(cls, n, point_type="matrix", epsilon=0.0, equip=True):
        """Instantiate a special orthogonal group.

        Select the object to instantiate depending on the point_type.
        """
        if n == 2 and point_type == "vector":
            return _SpecialOrthogonal2Vectors(epsilon, equip=equip)
        if n == 3 and point_type == "vector":
            return _SpecialOrthogonal3Vectors(epsilon, equip=equip)
        if point_type == "vector":
            raise NotImplementedError(
                "SO(n) is implemented in vector representation "
                "for n = 2 and n = 3 only."
            )
        return _SpecialOrthogonalMatrices(n, equip=equip)
