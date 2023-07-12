"""The special Euclidean group SE(n).

i.e. the Lie group of rigid transformations in n dimensions.

Lead authors: Nicolas Guigui and Nina Miolane.
"""
import math

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats.geometry.base import LevelSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.general_linear import GeneralLinear, Matrices
from geomstats.geometry.invariant_metric import InvariantMetric, _InvariantMetricMatrix
from geomstats.geometry.lie_algebra import MatrixLieAlgebra
from geomstats.geometry.lie_group import LieGroup, MatrixLieGroup
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.vectorization import repeat_out

PI = gs.pi
PI2 = PI * PI
PI3 = PI * PI2
PI4 = PI * PI3
PI5 = PI * PI4
PI6 = PI * PI5
PI7 = PI * PI6
PI8 = PI * PI7


ATOL = 1e-5


def _squared_dist_grad_point_a(point_a, point_b, metric):
    """Compute gradient of squared_dist wrt point_a.

    Compute the Riemannian gradient of the squared geodesic
    distance with respect to the first point point_a.

    Parameters
    ----------
    point_a : array-like, shape=[..., dim]
        Point.
    point_b : array-like, shape=[..., dim]
        Point.
    metric : SpecialEuclideanMatrixCanonicalLeftMetric
        Metric defining the distance.

    Returns
    -------
    _ : array-like, shape=[..., dim]
        Riemannian gradient, in the form of a tangent
        vector at base point : point_a.
    """
    return -2 * metric.log(point_b, point_a)


def _squared_dist_grad_point_b(point_a, point_b, metric):
    """Compute gradient of squared_dist wrt point_b.

    Compute the Riemannian gradient of the squared geodesic
    distance with respect to the second point point_b.

    Parameters
    ----------
    point_a : array-like, shape=[..., dim]
        Point.
    point_b : array-like, shape=[..., dim]
        Point.
    metric : SpecialEuclideanMatrixCanonicalLeftMetric
        Metric defining the distance.

    Returns
    -------
    _ : array-like, shape=[..., dim]
        Riemannian gradient, in the form of a tangent
        vector at base point : point_b.
    """
    return -2 * metric.log(point_a, point_b)


@gs.autodiff.custom_gradient(_squared_dist_grad_point_a, _squared_dist_grad_point_b)
def _squared_dist(point_a, point_b, metric):
    """Compute geodesic distance between two points.

    Compute the squared geodesic distance between point_a
    and point_b, as defined by the metric.

    This is an auxiliary private function that:

    - is called by the method `squared_dist` of the class
      SpecialEuclideanMatrixCanonicalLeftMetric,

    Parameters
    ----------
    point_a : array-like, shape=[..., dim]
        Point.
    point_b : array-like, shape=[..., dim]
        Point.
    metric : SpecialEuclideanMatrixCanonicalLeftMetric
        Metric defining the distance.

    Returns
    -------
    _ : array-like, shape=[...,]
        Geodesic distance between point_a and point_b.
    """
    return metric._squared_dist(point_a, point_b)


def homogeneous_representation(rotation, translation, constant=1.0):
    r"""Embed rotation, translation couples into n+1 square matrices.

    Construct a block matrix of size :math:`n + 1 \times n + 1` of the form

    .. math::
        \begin{pmatrix} R & t \\ 0 & c \end{pmatrix}

    where :math:`R` is a square matrix, :math:`t` a vector of size
    :math:`n`, and :math:`c` a constant (either 0 or 1 should be used).

    Parameters
    ----------
    rotation : array-like, shape=[..., n, n]
        Square Matrix.
    translation : array-like, shape=[..., n]
        Vector.
    constant : float or array-like of shape [...]
        Constant to use at the last line and column of the square matrix.
        Optional, default: 1.

    Returns
    -------
    mat: array-like, shape=[..., n + 1, n + 1]
        Square Matrix of size n + 1. It can represent an element of the
        special euclidean group or its Lie algebra.
    """
    if rotation.ndim > 2 or translation.ndim > 1:
        if rotation.ndim == 2:
            rotation = gs.broadcast_to(
                rotation, (translation.shape[0], *rotation.shape)
            )

        if translation.ndim == 1:
            translation = gs.broadcast_to(
                translation, (rotation.shape[0], *translation.shape)
            )

    mat = gs.concatenate((rotation, translation[..., None]), axis=-1)

    if not gs.is_array(constant) or constant.ndim == 0:
        constant = gs.array([constant])

    zeros = gs.zeros(mat.shape[:-1])
    if zeros.ndim > 1 or constant.shape[0] > 1:
        if zeros.ndim == 1:
            zeros = gs.broadcast_to(zeros, (constant.shape[0], *zeros.shape))

        if constant.shape[0] == 1:
            constant = gs.broadcast_to(constant, (zeros.shape[0], *constant.shape))
        else:
            constant = constant[..., None]

    last_row = gs.concatenate([zeros, constant], axis=-1)

    if mat.ndim == 2 and last_row.ndim > 1:
        mat = gs.broadcast_to(mat, (last_row.shape[0], *mat.shape))

    return gs.concatenate([mat, last_row[..., None, :]], axis=-2)


class _SpecialEuclideanMatrices(MatrixLieGroup, LevelSet):
    """Class for special Euclidean group.

    Parameters
    ----------
    n : int
        Integer dimension of the underlying Euclidean space. Matrices will
        be of size: (n+1) x (n+1).

    Attributes
    ----------
    rotations : SpecialOrthogonal
        Subgroup of rotations of size n.
    translations : Euclidean
        Subgroup of translations of size n.
    """

    def __init__(self, n, equip=True):
        self.n = n
        self._value = gs.eye(n + 1)

        super().__init__(
            dim=int((n * (n + 1)) / 2),
            representation_dim=n + 1,
            lie_algebra=SpecialEuclideanMatrixLieAlgebra(n=n),
            equip=equip,
        )
        self.rotations = SpecialOrthogonal(n=n, equip=True)
        self.translations = Euclidean(dim=n, equip=False)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return SpecialEuclideanMatrixCanonicalLeftMetric

    def _define_embedding_space(self):
        return GeneralLinear(self.n + 1, positive_det=True)

    def submersion(self, point):
        """Define SE(n) as the pre-image of 0.

        Parameters
        ----------
        point : array-like, shape=[..., n + 1, n + 1]
            Point.

        Returns
        -------
        submersed_point : array-like, shape=[..., n + 1, n + 1]
            Submersed Point.
        """
        n = self.n
        rot = point[..., :n, :n]
        vec = point[..., n, :n]
        scalar = point[..., n, n]
        submersed_rot = Matrices.mul(rot, Matrices.transpose(rot))
        return (
            homogeneous_representation(submersed_rot, vec, constant=scalar)
            - self._value
        )

    def tangent_submersion(self, vector, point):
        """Define the tangent space of SE(n) as the kernel of this method.

        Parameters
        ----------
        vector : array-like, shape=[..., n + 1, n + 1]
            Point.
        point : array-like, shape=[..., n + 1, n + 1]
            Point.

        Returns
        -------
        submersed_vector : array-like, shape=[..., n + 1, n + 1]
            Submersed Vector.
        """
        n = self.n
        rot = point[..., :n, :n]
        skew = vector[..., :n, :n]
        vec = vector[..., n, :n]
        scalar = vector[..., n, n]
        submersed_rot = Matrices.to_symmetric(
            Matrices.mul(Matrices.transpose(skew), rot)
        )
        return homogeneous_representation(submersed_rot, vec, constant=scalar)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in SE(n) from the product distribution.

        This method uses the distributions defined on the Euclidean and Special
        Orthogonal groups.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound: float
            Bound of the interval in which to sample each entry of the
            translation part.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n + 1, n + 1]
            Sample in SE(n).
        """
        random_translation = self.translations.random_point(n_samples)
        random_rotation = self.rotations.random_uniform(n_samples)
        random_point = homogeneous_representation(random_rotation, random_translation)
        return random_point

    @classmethod
    def inverse(cls, point):
        """Return the inverse of a point.

        Parameters
        ----------
        point : array-like, shape=[..., n + 1, n + 1]
            Point to be inverted.

        Returns
        -------
        inverse : array-like, shape=[..., n + 1, n + 1]
            Inverse of point.
        """
        n = point.shape[-1] - 1
        transposed_rot = Matrices.transpose(point[..., :n, :n])
        translation = point[..., :n, -1]
        translation = gs.einsum("...ij,...j->...i", transposed_rot, translation)
        return homogeneous_representation(transposed_rot, -translation)

    def projection(self, mat):
        """Project a matrix on SE(n).

        The upper-left n x n block is projected to SO(n) by minimizing the
        Frobenius norm. The last columns is kept unchanged and used as the
        translation part. The last row is discarded.

        Parameters
        ----------
        mat : array-like, shape=[..., n + 1, n + 1]
            Matrix.

        Returns
        -------
        projected : array-like, shape=[..., n + 1, n + 1]
            Rotation-translation matrix in homogeneous representation.
        """
        if gs.any(self.belongs(mat)):
            # otherwise, there will be problems with autodiff
            return gs.copy(mat)

        n = self.n
        projected_rot = self.rotations.projection(mat[..., :n, :n])
        translation = mat[..., :n, -1]
        return homogeneous_representation(projected_rot, translation)


class _SpecialEuclideanVectors(LieGroup):
    """Base Class for the special Euclidean groups in 2d and 3d in vector form.

    i.e. the Lie group of rigid transformations. Elements of SE(2), SE(3) can
    either be represented as vectors (in 2d or 3d) or as matrices in general.
    The matrix representation corresponds to homogeneous coordinates. This
    class is specific to the vector representation of rotations. For the matrix
    representation use the SpecialEuclidean class and set `n=2` or `n=3`.

    Parameter
    ---------
    epsilon : float
        Precision to use for calculations involving potential
        division by 0 in rotations.
        Optional, default: 0.
    """

    def __init__(self, n, epsilon=0.0, equip=True):
        self.n = n
        self.epsilon = epsilon
        self.rotations = SpecialOrthogonal(
            n=n, point_type="vector", epsilon=epsilon, equip=False
        )
        self.translations = Euclidean(dim=n, equip=False)

        dim = n * (n + 1) // 2
        super().__init__(
            dim=dim,
            shape=(dim,),
            lie_algebra=Euclidean(dim),
            equip=equip,
        )

    @property
    def identity(self):
        return gs.zeros(self.dim)

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to SE(2) or SE(3).

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to check.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean indicating whether point belongs to SE(2) or SE(3).
        """
        point_dim = point.shape[-1]
        point_ndim = point.ndim
        belongs = gs.logical_and(point_dim == self.dim, point_ndim < 3)
        belongs = gs.logical_and(
            belongs, self.rotations.belongs(point[..., : self.rotations.dim], atol=atol)
        )
        return belongs

    def projection(self, point):
        """Project a point to the group.

        The point is regularized, so that the norm of the rotation part lie in [0, pi).

        Parameters
        ----------
        point: array-like, shape[..., dim]
            Point.

        Returns
        -------
        projected: array-like, shape[..., dim]
            Regularized point.
        """
        return self.regularize(point)

    def regularize(self, point):
        """Regularize a point to the default representation for SE(n).

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to regularize.

        Returns
        -------
        point : array-like, shape=[..., dim]
            Regularized point.
        """
        rotations = self.rotations
        dim_rotations = rotations.dim

        regularized_point = gs.copy(point)
        rot_vec = regularized_point[..., :dim_rotations]
        regularized_rot_vec = rotations.regularize(rot_vec)

        translation = regularized_point[..., dim_rotations:]

        return gs.concatenate([regularized_rot_vec, translation], axis=-1)

    def regularize_tangent_vec_at_identity(self, tangent_vec):
        """Regularize a tangent vector at the identity.

        Parameters
        ----------
        tangent_vec: array-like, shape=[..., dim]
            Tangent vector at base point.

        Returns
        -------
        regularized_vec : array-like, shape=[..., dim]
            Regularized vector.
        """
        return self.regularize_tangent_vec(tangent_vec, self.identity)

    def matrix_from_vector(self, vec):
        """Convert point in vector point-type to matrix.

        Parameters
        ----------
        vec : array-like, shape=[..., dim]
            Vector.

        Returns
        -------
        mat : array-like, shape=[..., n+1, n+1]
            Matrix.
        """
        vec = self.regularize(vec)
        rot_vec = vec[..., : self.rotations.dim]
        trans_vec = vec[..., self.rotations.dim :]

        rot_mat = self.rotations.matrix_from_rotation_vector(rot_vec)
        return homogeneous_representation(rot_mat, trans_vec)

    def compose(self, point_a, point_b):
        r"""Compose two elements of SE(2) or SE(3).

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point of the group.
        point_b : array-like, shape=[..., dim]
            Point of the group.

        Equation
        --------
        (:math:`(R_1, t_1) \\cdot (R_2, t_2) = (R_1 R_2, R_1 t_2 + t_1)`)

        Returns
        -------
        composition : array-like, shape=[..., dim]
            Composition of point_a and point_b.
        """
        rotations = self.rotations
        dim_rotations = rotations.dim

        point_a = self.regularize(point_a)
        point_b = self.regularize(point_b)

        rot_vec_a = point_a[..., :dim_rotations]
        rot_mat_a = rotations.matrix_from_rotation_vector(rot_vec_a)

        rot_vec_b = point_b[..., :dim_rotations]
        rot_mat_b = rotations.matrix_from_rotation_vector(rot_vec_b)

        translation_a = point_a[..., dim_rotations:]
        translation_b = point_b[..., dim_rotations:]

        composition_rot_mat = gs.matmul(rot_mat_a, rot_mat_b)
        composition_rot_vec = rotations.rotation_vector_from_matrix(composition_rot_mat)

        composition_translation = (
            gs.einsum("...j,...kj->...k", translation_b, rot_mat_a) + translation_a
        )

        composition = gs.concatenate(
            (composition_rot_vec, composition_translation), axis=-1
        )
        return self.regularize(composition)

    def inverse(self, point):
        r"""Compute the group inverse in SE(n).

        Parameters
        ----------
        point: array-like, shape=[..., dim]
            Point.

        Returns
        -------
        inverse_point : array-like, shape=[..., dim]
            Inverted point.

        Notes
        -----
        :math:`(R, t)^{-1} = (R^{-1}, R^{-1}.(-t))`
        """
        rotations = self.rotations
        dim_rotations = rotations.dim

        point = self.regularize(point)

        rot_vec = point[..., :dim_rotations]
        translation = point[..., dim_rotations:]

        inverse_rotation = -rot_vec

        inv_rot_mat = rotations.matrix_from_rotation_vector(inverse_rotation)

        inverse_translation = gs.einsum("...i,...ji->...j", -translation, inv_rot_mat)

        inverse_point = gs.concatenate([inverse_rotation, inverse_translation], axis=-1)
        return self.regularize(inverse_point)

    def exp_from_identity(self, tangent_vec):
        """Compute group exponential of the tangent vector at the identity.

        Parameters
        ----------
        tangent_vec: array-like, shape=[..., dim]
            Tangent vector at base point.

        Returns
        -------
        group_exp: array-like, shape=[..., dim]
            Group exponential of the tangent vectors computed
            at the identity.
        """
        rotations = self.rotations
        dim_rotations = rotations.dim

        rot_vec = tangent_vec[..., :dim_rotations]
        rot_vec_regul = self.rotations.regularize(rot_vec)

        transform = self._exp_translation_transform(rot_vec_regul)

        translation = tangent_vec[..., dim_rotations:]
        exp_translation = gs.einsum("...jk,...k->...j", transform, translation)

        group_exp = gs.concatenate([rot_vec, exp_translation], axis=-1)

        group_exp = self.regularize(group_exp)
        return group_exp

    def log_from_identity(self, point):
        """Compute the group logarithm of the point at the identity.

        Parameters
        ----------
        point: array-like, shape=[..., 3]
            Point.

        Returns
        -------
        group_log: array-like, shape=[..., 3]
            Group logarithm in the Lie algebra.
        """
        point = self.regularize(point)

        rotations = self.rotations
        dim_rotations = rotations.dim

        rot_vec = point[..., :dim_rotations]
        translation = point[..., dim_rotations:]

        transform = self._log_translation_transform(rot_vec)
        log_translation = gs.einsum("...jk, ...k -> ...j", transform, translation)

        return gs.concatenate([rot_vec, log_translation], axis=-1)

    def random_point(self, n_samples=1, bound=1.0, **kwargs):
        """Sample in SE(n) from the product distribution.

        This method uses the distributions defined on the Euclidean and Special
        Orthogonal groups.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Upper bound for the translation part of the sample.
            Optional, default: 1.

        Returns
        -------
        random_point : array-like, shape=[..., dim]
            Sample.
        """
        random_translation = self.translations.random_point(n_samples, bound)
        random_rot_vec = self.rotations.random_uniform(n_samples)
        return gs.concatenate([random_rot_vec, random_translation], axis=-1)


class _SpecialEuclidean2Vectors(_SpecialEuclideanVectors):
    """Class for the special Euclidean group in 2d, SE(2).

    i.e. the Lie group of rigid transformations. Elements of SE(32 can either
    be represented as vectors (in 2d) or as matrices in general. The matrix
    representation corresponds to homogeneous coordinates. This class is
    specific to the vector representation of rotations. For the matrix
    representation use the SpecialEuclidean class and set `n=2`.

    Parameter
    ---------
    epsilon : float
        Precision to use for calculations involving potential
        division by 0 in rotations.
        Optional, default: 0.
    """

    def __init__(self, epsilon=0.0, equip=True):
        super().__init__(n=2, epsilon=epsilon, equip=equip)

    def regularize_tangent_vec(self, tangent_vec, base_point):
        """Regularize a tangent vector at a base point.

        Parameters
        ----------
        tangent_vec: array-like, shape=[..., 3]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 3]
            Base point.

        Returns
        -------
        regularized_vec : array-like, shape=[..., 3]
            Regularized vector.
        """
        rotations = self.rotations
        dim_rotations = rotations.dim

        rot_tangent_vec = tangent_vec[..., :dim_rotations]
        rot_base_point = base_point[..., :dim_rotations]

        rotations_vec = rotations.regularize_tangent_vec(
            tangent_vec=rot_tangent_vec, base_point=rot_base_point
        )

        return gs.concatenate(
            [rotations_vec, tangent_vec[..., dim_rotations:]], axis=-1
        )

    def jacobian_translation(self, point, left=True):
        """Compute the Jacobian matrix resulting from translation.

        Compute the matrix of the differential of the left/right translations
        from the identity to point in SE(3).

        Parameters
        ----------
        point: array-like, shape=[..., 3]
            Point.
        left: bool
            Whether to compute the jacobian of the left or right translation.
            Optional, default: True.

        Returns
        -------
        jacobian : array-like, shape=[..., 3]
            Jacobian of the left / right translation.
        """
        n_points = point.shape[0] if gs.ndim(point) > 1 else 1
        out = gs.eye(self.dim)

        if n_points > 1:
            return gs.repeat(gs.expand_dims(out, axis=0), n_points, axis=0)
        return out

    def _exp_translation_transform(self, rot_vec):
        base_1 = gs.eye(2)
        base_2 = self.rotations.skew_matrix_from_vector(gs.ones(1))
        cos_coef = rot_vec * utils.taylor_exp_even_func(
            rot_vec**2, utils.cosc_close_0, order=3
        )
        sin_coef = utils.taylor_exp_even_func(rot_vec**2, utils.sinc_close_0, order=3)

        sin_term = gs.einsum("...i,...jk->...jk", sin_coef, base_1)
        cos_term = gs.einsum("...i,...jk->...jk", cos_coef, base_2)
        transform = sin_term + cos_term

        return transform

    def _log_translation_transform(self, rot_vec):
        exp_transform = self._exp_translation_transform(rot_vec)

        inv_determinant = 0.5 / utils.taylor_exp_even_func(
            rot_vec**2, utils.cosc_close_0, order=4
        )
        transform = gs.einsum(
            "...l, ...jk -> ...jk", inv_determinant, Matrices.transpose(exp_transform)
        )

        return transform


class _SpecialEuclidean3Vectors(_SpecialEuclideanVectors):
    """Class for the special Euclidean group in 3d, SE(3).

    i.e. the Lie group of rigid transformations. Elements of SE(3) can either
    be represented as vectors (in 3d) or, in general, as matrices. The matrix
    representation corresponds to homogeneous coordinates. This class is
    specific to the vector representation of rotations. For the matrix
    representation use the SpecialEuclidean class and set `n=3`.

    Parameter
    ---------
    epsilon : float
        Precision to use for calculations involving potential
        division by 0 in rotations.
        Optional, default: 0.
    """

    def __init__(self, epsilon=0.0, equip=True):
        super().__init__(n=3, epsilon=epsilon, equip=equip)

    def equip_with_metric(self, Metric=None, **metric_kwargs):
        super().equip_with_metric(Metric=Metric, **metric_kwargs)

        dim_rotations = self.rotations.dim
        metric_mat = self.metric.metric_mat_at_identity
        rot_metric_mat = metric_mat[:dim_rotations, :dim_rotations]
        rotations_kwargs = {
            "metric_mat_at_identity": rot_metric_mat,
            "left": self.metric.left,
        }
        self.rotations.equip_with_metric(InvariantMetric, **rotations_kwargs)

    def regularize_tangent_vec(self, tangent_vec, base_point):
        """Regularize a tangent vector at a base point.

        Parameters
        ----------
        tangent_vec: array-like, shape=[..., 3]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 3]
            Base point.

        Returns
        -------
        regularized_vec : array-like, shape=[..., 3]
            Regularized vector.
        """
        dim_rotations = self.rotations.dim

        rot_tangent_vec = tangent_vec[..., :dim_rotations]
        rot_base_point = base_point[..., :dim_rotations]

        rotations_vec = self.rotations.regularize_tangent_vec(
            tangent_vec=rot_tangent_vec,
            base_point=rot_base_point,
        )

        return gs.concatenate(
            [rotations_vec, tangent_vec[..., dim_rotations:]], axis=-1
        )

    def jacobian_translation(self, point, left=True):
        """Compute the Jacobian matrix resulting from translation.

        Compute the matrix of the differential of the left/right translations
        from the identity to point in SE(3).

        Parameters
        ----------
        point: array-like, shape=[..., 3]
            Point.
        left: bool
            Whether to compute the jacobian of the left or right translation.
            Optional, default: True.

        Returns
        -------
        jacobian : array-like, shape=[..., 3]
            Jacobian of the left / right translation.
        """
        rotations = self.rotations
        translations = self.translations
        dim_rotations = rotations.dim
        dim_translations = translations.dim

        n_points = point.shape[0] if gs.ndim(point) > 1 else 1
        is_vec = gs.ndim(point) > 1
        n_points_shape = (n_points,) if is_vec else ()

        point = self.regularize(point)
        rot_vec = point[..., :dim_rotations]

        jacobian_rot = self.rotations.jacobian_translation(point=rot_vec, left=left)

        block_zeros_1 = gs.zeros(n_points_shape + (dim_rotations, dim_translations))
        jacobian_block_line_1 = gs.concatenate([jacobian_rot, block_zeros_1], axis=-1)

        if left:
            jacobian_trans = self.rotations.matrix_from_rotation_vector(rot_vec)

            block_zeros_2 = gs.zeros(n_points_shape + (dim_translations, dim_rotations))
            jacobian_block_line_2 = gs.concatenate(
                [block_zeros_2, jacobian_trans], axis=-1
            )

        else:
            inv_skew_mat = -self.rotations.skew_matrix_from_vector(rot_vec)
            eye = gs.eye(self.n)
            if is_vec:
                eye = gs.repeat(gs.expand_dims(eye, axis=0), n_points, axis=0)
            jacobian_block_line_2 = gs.concatenate([inv_skew_mat, eye], axis=-1)

        jacobian = gs.concatenate(
            [jacobian_block_line_1, jacobian_block_line_2], axis=-2
        )
        return jacobian

    def _exponential_matrix(self, rot_vec):
        """Compute exponential of rotation matrix represented by rot_vec.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]

        Returns
        -------
        exponential_mat : Matrix exponential of rot_vec
        """
        # TODO (nguigs): find usecase for this method
        rot_vec = self.rotations.regularize(rot_vec)
        n_rot_vecs = 1 if rot_vec.ndim == 1 else len(rot_vec)

        angle = gs.linalg.norm(rot_vec, axis=-1)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

        skew_rot_vec = self.rotations.skew_matrix_from_vector(rot_vec)

        coef_1 = gs.empty_like(angle)
        coef_2 = gs.empty_like(coef_1)

        mask_0 = gs.equal(angle, 0)
        mask_0 = gs.squeeze(mask_0, axis=1)
        mask_close_to_0 = gs.isclose(angle, 0)
        mask_close_to_0 = gs.squeeze(mask_close_to_0, axis=1)
        mask_else = ~mask_0 & ~mask_close_to_0

        coef_1[mask_close_to_0] = 1.0 / 2.0 - angle[mask_close_to_0] ** 2 / 24.0
        coef_2[mask_close_to_0] = 1.0 / 6.0 - angle[mask_close_to_0] ** 3 / 120.0

        # TODO (nina): Check if the discontinuity at 0 is expected.
        coef_1[mask_0] = 0
        coef_2[mask_0] = 0

        coef_1[mask_else] = angle[mask_else] ** (-2) * (1.0 - gs.cos(angle[mask_else]))
        coef_2[mask_else] = angle[mask_else] ** (-2) * (
            1.0 - (gs.sin(angle[mask_else]) / angle[mask_else])
        )

        term_1 = gs.zeros((n_rot_vecs, self.n, self.n))
        term_2 = gs.zeros_like(term_1)

        for i in range(n_rot_vecs):
            term_1[i] = gs.eye(self.n) + skew_rot_vec[i] * coef_1[i]
            term_2[i] = gs.matmul(skew_rot_vec[i], skew_rot_vec[i]) * coef_2[i]

        exponential_mat = term_1 + term_2

        return exponential_mat

    def _exp_translation_transform(self, rot_vec):
        """Compute matrix associated to rot_vec for the translation part in exp.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]

        Returns
        -------
        transform : array-like, shape=[..., 3, 3]
            Matrix to be applied to the translation part in exp.
        """
        sq_angle = gs.sum(rot_vec**2, axis=-1)
        skew_mat = self.rotations.skew_matrix_from_vector(rot_vec)
        sq_skew_mat = gs.matmul(skew_mat, skew_mat)

        coef_1_ = utils.taylor_exp_even_func(sq_angle, utils.cosc_close_0, order=4)
        coef_2_ = utils.taylor_exp_even_func(sq_angle, utils.var_sinc_close_0, order=4)

        term_1 = gs.einsum("...,...ij->...ij", coef_1_, skew_mat)
        term_2 = gs.einsum("...,...ij->...ij", coef_2_, sq_skew_mat)
        term_id = gs.eye(3)
        transform = term_id + term_1 + term_2

        return transform

    def _log_translation_transform(self, rot_vec):
        """Compute matrix associated to rot_vec for the translation part in log.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]

        Returns
        -------
        transform : array-like, shape=[..., 3, 3]
        Matrix to be applied to the translation part in log
        """
        angle = gs.linalg.norm(rot_vec, axis=-1)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=-1)

        skew_mat = self.rotations.skew_matrix_from_vector(rot_vec)
        sq_skew_mat = gs.matmul(skew_mat, skew_mat)

        mask_close_0 = gs.isclose(angle, 0.0)
        mask_close_pi = gs.isclose(angle, gs.pi)
        mask_else = ~mask_close_0 & ~mask_close_pi

        mask_close_0_float = gs.cast(mask_close_0, rot_vec.dtype)
        mask_close_pi_float = gs.cast(mask_close_pi, rot_vec.dtype)
        mask_else_float = gs.cast(mask_else, rot_vec.dtype)

        mask_0 = gs.isclose(angle, 0.0, atol=1e-7)
        mask_0_float = gs.cast(mask_0, rot_vec.dtype)
        angle += mask_0_float * gs.ones_like(angle)

        coef_1 = -0.5 * gs.ones_like(angle)
        coef_2 = gs.zeros_like(angle)

        coef_2 += mask_close_0_float * (
            1.0 / 12.0
            + angle**2 / 720.0
            + angle**4 / 30240.0
            + angle**6 / 1209600.0
        )

        delta_angle = angle - gs.pi
        coef_2 += mask_close_pi_float * (
            1.0 / PI2
            + (PI2 - 8.0) * delta_angle / (4.0 * PI3)
            - ((PI2 - 12.0) * delta_angle**2 / (4.0 * PI4))
            + ((-192.0 + 12.0 * PI2 + PI4) * delta_angle**3 / (48.0 * PI5))
            - ((-240.0 + 12.0 * PI2 + PI4) * delta_angle**4 / (48.0 * PI6))
            + (
                (-2880.0 + 120.0 * PI2 + 10.0 * PI4 + PI6)
                * delta_angle**5
                / (480.0 * PI7)
            )
            - (
                (-3360 + 120.0 * PI2 + 10.0 * PI4 + PI6)
                * delta_angle**6
                / (480.0 * PI8)
            )
        )

        psi = 0.5 * angle * gs.sin(angle) / (1 - gs.cos(angle))
        coef_2 += mask_else_float * (1 - psi) / (angle**2)

        term_1 = gs.einsum("...,...j->...j", coef_1, skew_mat)
        term_2 = gs.einsum("...,...j->...j", coef_2, sq_skew_mat)
        term_id = gs.eye(3)
        transform = term_id + term_1 + term_2

        return transform


class SpecialEuclideanMatrixCanonicalLeftMetric(_InvariantMetricMatrix):
    """Class for the canonical left-invariant metric on SE(n).

    The canonical left-invariant metric is defined by endowing the tangent
    space at the identity with the Frobenius inned-product, and to define the
    metric at any point by left-translation. This results in a direct product
    metric between rotations and translations, whose geodesics are therefore
    easily computable with the matrix exponential and straight lines.

    Parameters
    ----------
    space : SpecialEuclidean
        Instance of the class SpecialEuclidean with `point_type='matrix'`.
    """

    def __init__(self, space):
        if not self._check_implemented(space):
            raise ValueError(
                "group must be an instance of the "
                "SpecialEuclidean class with `point_type=matrix`."
            )
        super().__init__(space=space)

    def _instantiate_solvers(self):
        pass

    def _check_implemented(self, space):
        return (
            isinstance(space, _SpecialEuclideanMatrices)
            and space.default_point_type == "matrix"
        )

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute inner product of two vectors in tangent space at base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            First tangent vector at base_point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Second tangent vector at base_point.
        base_point : array-like, shape=[..., n, n]
            Point in the group.
            Optional, defaults to identity if None.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        inner_prod = Matrices.frobenius_product(tangent_vec_a, tangent_vec_b)
        return repeat_out(
            self._space, inner_prod, base_point, tangent_vec_a, tangent_vec_b
        )

    def exp(self, tangent_vec, base_point=None):
        """Exponential map associated to the cannonical metric.

        Exponential map at `base_point` of `tangent_vec`. The geodesics of this
        metric correspond to a direct product metric between rotation and
        translation: the translation part is a straight line, while the
        rotation part has constant angular velocity (which corresponds to one-
        parameter subgroups of the rotation group).

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n + 1, n + 1]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., n + 1, n + 1]
            Point on the manifold.

        Returns
        -------
        exp : array-like, shape=[..., n + 1, n + 1]
            Point on the manifold.

        See Also
        --------
        examples.plot_geodesics_se2
        """
        group = self._space
        n = group.n
        if base_point is None:
            base_point = group.identity
        inf_rotation = tangent_vec[..., :n, :n]
        rotation = base_point[..., :n, :n]
        rotation_exp = GeneralLinear.exp(inf_rotation, rotation)
        translation_exp = tangent_vec[..., :n, n] + base_point[..., :n, n]

        return homogeneous_representation(rotation_exp, translation_exp, 1.0)

    def log(self, point, base_point=None):
        """Compute logarithm map associated to the canonical metric.

        Log map at `base_point` of `point`. The geodesics of this
        metric correspond to a direct product metric between rotation and
        translation: the translation part is a straight line, while the
        rotation part has constant angular velocity (which corresponds to one-
        parameter subgroups of the rotation group).

        Parameters
        ----------
        point : array-like, shape=[..., n + 1, n + 1]
            Point on the manifold.
        base_point : array-like, shape=[..., n + 1, n + 1]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n + 1, n + 1]
            Tangent vector at the base point.

        References
        ----------
        .. [Zefran98] Zefran, M., V. Kumar, and C.B. Croke.
            “On the Generation of Smooth Three-Dimensional Rigid Body Motions.”
            IEEE Transactions on Robotics and Automation 14,
            no. 4 (August 1998): 576–89.
            https://doi.org/10.1109/70.704225.
        """
        n = self._space.n
        rotation_bp = base_point[..., :n, :n]
        rotation_p = point[..., :n, :n]
        rotation_log = GeneralLinear.log(rotation_p, rotation_bp)
        translation_log = point[..., :n, n] - base_point[..., :n, n]

        return homogeneous_representation(rotation_log, translation_log, 0.0)

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        """Generate parameterized function for the geodesic curve.

        Geodesic curve defined by either:

        - an initial point and an initial tangent vector,
        - an initial point and an end point.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Point on the manifold, initial point of the geodesic.
        end_point : array-like, shape=[..., dim], optional
            Point on the manifold, end point of the geodesic. If None,
            an initial tangent vector must be given.
        initial_tangent_vec : array-like, shape=[..., dim],
            Tangent vector at base point, the initial speed of the geodesics.
            Optional, default: None.
            If None, an end point must be given and a logarithm is computed.

        Returns
        -------
        path : callable
            Time parameterized geodesic curve. If a batch of initial
            conditions is passed, the output array's first dimension
            represents the different initial conditions, and the second
            corresponds to time.
        """
        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end point or an initial tangent "
                "vector to define the geodesic."
            )
        if end_point is not None:
            if initial_tangent_vec is not None:
                raise ValueError(
                    "Cannot specify both an end point and an initial tangent vector."
                )
            initial_tangent_vec = self.log(end_point, initial_point)

        return self._geodesic_from_exp(initial_point, initial_tangent_vec)

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None, **kwargs
    ):
        r"""Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector a
        along the geodesic between two points `base_point` and `end_point`
        or alternatively defined by :math:`t \mapsto exp_{(base\_point)}(
        t*direction)`. As the special Euclidean group endowed with its
        canonical left-invariant metric is a symmetric space, parallel
        transport is achieved by a geodesic symmetry, or equivalently, one step
        of the pole ladder scheme.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n + 1, n + 1]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., n + 1, n + 1]
            Point on the hypersphere.
        direction : array-like, shape=[..., n + 1, n + 1]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default: None
        end_point : array-like, shape=[..., n + 1, n + 1]
            Point on the Grassmann manifold to transport to. Unused if
            `tangent_vec_b` is given.
            Optional, default: None

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., n + 1, n + 1]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        """
        n = self._space.n
        if direction is None:
            if end_point is not None:
                direction = self.log(end_point, base_point)
            else:
                raise ValueError(
                    "Either an end_point or a tangent_vec_b must be given to define the"
                    " geodesic along which to transport."
                )
        rot_a = tangent_vec[..., :n, :n]
        rot_b = direction[..., :n, :n]
        rot_bp = base_point[..., :n, :n]
        transported_rot = self._space.rotations.metric.parallel_transport(
            rot_a, rot_bp, rot_b
        )
        translation = tangent_vec[..., :n, n]

        return homogeneous_representation(transported_rot, translation, 0.0)

    def _squared_dist(self, point_a, point_b):
        """Compute geodesic distance between two points.

        Compute the squared geodesic distance between point_a
        and point_b, as defined by the metric.

        This is an auxiliary private function that:

        - is called by the method `squared_dist` of the class
          SpecialEuclideanMatrixCanonicalLeftMetric,

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point.
        point_b : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        _ : array-like, shape=[...,]
            Geodesic distance between point_a and point_b.
        """
        return super().squared_dist(point_a, point_b)

    def squared_dist(self, point_a, point_b, **kwargs):
        """Squared geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point.
        point_b : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
            Squared distance.
        """
        return _squared_dist(point_a, point_b, metric=self)

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base point onto
        its image.
        In this case, it does not depend on the base point. If the rotation part is
        null, then the radius is infinite, otherwise it is the same as the special
        orthonormal group.

        Parameters
        ----------
        base_point : array-like, shape=[..., n + 1, n + 1]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        n = self._space.n
        rotation = base_point[..., :n, :n]
        rotation_radius = gs.pi * (self._space.dim - n) ** 0.5
        return gs.where(gs.sum(rotation, axis=(-2, -1)) == 0, math.inf, rotation_radius)


class SpecialEuclidean:
    r"""Class for the special Euclidean groups.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices : n x n.
    point_type : str, {\'vector\', \'matrix\'}
        Representation of the elements of the group.
        Optional, default: 'matrix',
    epsilon : float
        Precision used for calculations involving potential divison by 0 in
        rotations.
        Optional, default: 0.
    """

    def __new__(cls, n, point_type="matrix", epsilon=0.0, equip=True):
        """Instantiate a special Euclidean group.

        Select the object to instantiate depending on the point_type.
        """
        if n == 2 and point_type == "vector":
            return _SpecialEuclidean2Vectors(epsilon, equip=equip)
        if n == 3 and point_type == "vector":
            return _SpecialEuclidean3Vectors(epsilon, equip=equip)
        if point_type == "vector":
            raise NotImplementedError(
                "SE(n) is only implemented in matrix representation when n > 3."
            )
        return _SpecialEuclideanMatrices(n, equip=equip)


class SpecialEuclideanMatrixLieAlgebra(MatrixLieAlgebra):
    r"""Lie Algebra of the special Euclidean group.

    This is the tangent space at the identity. It is identified with the
    :math:`n + 1 \times n + 1` block matrices of the form:

    .. math::
        ((A, t), (0, 0))

    where A is an :math:`n \times n` skew-symmetric matrix, :math:`t` is an
    n-dimensional vector.

    Parameters
    ----------
    n : int
        Integer dimension of the underlying Euclidean space. Matrices will
        be of size: (n+1) x (n+1).
    """

    def __init__(self, n):
        self.n = n
        dim = int(n * (n + 1) / 2)
        super().__init__(dim=dim, representation_dim=n + 1, equip=False)

        self.skew = SkewSymmetricMatrices(n)

    def _create_basis(self):
        """Create the canonical basis."""
        n = self.n
        basis = homogeneous_representation(
            self.skew.basis,
            gs.zeros((self.skew.dim, n)),
            0.0,
        )

        indices = [(row, row, n) for row in range(n)]
        add_basis = gs.array_from_sparse(indices, [1.0] * n, (n, n + 1, n + 1))

        return gs.vstack([basis, add_basis])

    def belongs(self, point, atol=ATOL):
        """Evaluate if the rotation part of point is a skew-symmetric matrix.

        Parameters
        ----------
        point : array-like, shape=[..., n + 1, n + 1]
            Square matrix to check.
        atol : float
            Tolerance for the equality evaluation.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if rotation part of matrix is skew symmetric.
        """
        point_dim1, point_dim2 = point.shape[-2:]
        belongs = point_dim1 == point_dim2 == self.n + 1

        rotation = point[..., : self.n, : self.n]
        rot_belongs = self.skew.belongs(rotation, atol=atol)

        belongs = gs.logical_and(belongs, rot_belongs)

        last_line = point[..., -1, :]
        all_zeros = ~gs.any(last_line, axis=-1)

        belongs = gs.logical_and(belongs, all_zeros)
        return belongs

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the lie algebra with a uniform distribution in a box.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0

        Returns
        -------
        point : array-like, shape=[..., n + 1, n + 1]
           Sample.
        """
        point = super().random_point(n_samples, bound)
        return self.projection(point)

    def projection(self, mat):
        """Project a matrix to the Lie Algebra.

        Compute the skew-symmetric projection of the rotation part of matrix.

        Parameters
        ----------
        mat : array-like, shape=[..., n + 1, n + 1]
            Matrix.

        Returns
        -------
        projected : array-like, shape=[..., n + 1, n + 1]
            Matrix belonging to Lie Algebra.
        """
        rotation = mat[..., : self.n, : self.n]
        skew = SkewSymmetricMatrices.projection(rotation)
        return homogeneous_representation(skew, mat[..., : self.n, self.n], 0.0)

    def basis_representation(self, matrix_representation):
        """Calculate the coefficients of given matrix in the basis.

        Compute a 1d-array that corresponds to the input matrix in the basis
        representation.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n + 1, n + 1]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., dim]
            Representation in the basis.
        """
        skew_part = self.skew.basis_representation(
            matrix_representation[..., : self.n, : self.n]
        )
        translation_part = matrix_representation[..., :-1, self.n]
        return gs.concatenate([skew_part, translation_part[..., :]], axis=-1)
