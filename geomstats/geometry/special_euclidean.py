"""The special Euclidean group SE(n).

i.e. the Lie group of rigid transformations in n dimensions.
"""

import geomstats.algebra_utils as utils
import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry.base import EmbeddedManifold
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.general_linear import GeneralLinear, Matrices
from geomstats.geometry.invariant_metric import _InvariantMetricMatrix
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.lie_algebra import MatrixLieAlgebra
from geomstats.geometry.lie_group import LieGroup, MatrixLieGroup
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

PI = gs.pi
PI2 = PI * PI
PI3 = PI * PI2
PI4 = PI * PI3
PI5 = PI * PI4
PI6 = PI * PI5
PI7 = PI * PI6
PI8 = PI * PI7


ATOL = 1e-5

TAYLOR_COEFFS_1_AT_0 = [+ 1. / 2., 0.,
                        - 1. / 24., 0.,
                        + 1. / 720., 0.,
                        - 1. / 40320.]

TAYLOR_COEFFS_2_AT_0 = [+ 1. / 6., 0.,
                        - 1. / 120., 0.,
                        + 1. / 5040., 0.,
                        - 1. / 362880.]


def homogeneous_representation(
        rotation, translation, output_shape, constant=1.):
    r"""Embed rotation, translation couples into n+1 square matrices.

    Construct a block matrix of size :math: `n + 1 \times n + 1` of the form
    .. math::
        \matvec{cc}{R & t\\
                    0&c}

    where :math: `R` is a square matrix, :math: `t` a vector of size
    :math: `n`, and :math: `c` a constant (either 0 or 1 should be used).

    Parameters
    ----------
    rotation : array-like, shape=[..., n, n]
        Square Matrix.
    translation : array-like, shape=[..., n]
        Vector.
    output_shape : tuple of int
        Desired output shape. This is need for vectorization.
    constant : float or array-like of shape [...]
        Constant to use at the last line and column of the square matrix.
        Optional, default: 1.

    Returns
    -------
    mat: array-like, shape=[..., n + 1, n + 1]
        Square Matrix of size n + 1. It can represent an element of the
        special euclidean group or its Lie algebra.
    """
    mat = gs.concatenate((rotation, translation[..., None]), axis=-1)
    last_line = gs.zeros(output_shape)[..., -1]
    if isinstance(constant, float):
        last_col = constant * gs.ones_like(translation)[..., None, -1]
    else:
        last_col = constant[..., None]
    last_line = gs.concatenate(
        [last_line[..., :-1], last_col], axis=-1)
    mat = gs.concatenate((mat, last_line[..., None, :]), axis=-2)
    return mat


def submersion(point):
    """Define SE(n) as the pre-image of identity.

    Parameters
    ----------
    point : array-like, shape=[..., n + 1, n + 1]
        Point.

    Returns
    -------
    submersed_point : array-like, shape=[..., n + 1, n + 1]
        Submersed Point.
    """
    n = point.shape[-1] - 1
    rot = point[..., :n, :n]
    vec = point[..., n, :n]
    scalar = point[..., n, n]
    submersed_rot = Matrices.mul(rot, Matrices.transpose(rot))
    return homogeneous_representation(
        submersed_rot, vec, point.shape, constant=scalar)


def tangent_submersion(vector, point):
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
    n = point.shape[-1] - 1
    rot = point[..., :n, :n]
    skew = vector[..., :n, :n]
    vec = vector[..., n, :n]
    scalar = vector[..., n, n]
    submersed_rot = Matrices.mul(Matrices.transpose(skew), rot)
    submersed_rot = Matrices.to_symmetric(submersed_rot)
    return homogeneous_representation(
        submersed_rot, vec, point.shape, constant=scalar)


class _SpecialEuclideanMatrices(MatrixLieGroup, EmbeddedManifold):
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
    left_canonical_metric : InvariantMetric
        The left invariant metric that corresponds to the Frobenius inner
        product at the identity.
    right_canonical_metric : InvariantMetric
        The right invariant metric that corresponds to the Frobenius inner
        product at the identity.
    metric :  MatricesMetric
        The Euclidean (Frobenius) inner product.
    """

    def __init__(self, n):
        super().__init__(
            n=n + 1, dim=int((n * (n + 1)) / 2),
            embedding_space=GeneralLinear(n + 1, positive_det=True),
            submersion=submersion, value=gs.eye(n + 1),
            tangent_submersion=tangent_submersion,
            lie_algebra=SpecialEuclideanMatrixLieAlgebra(n=n))
        self.rotations = SpecialOrthogonal(n=n)
        self.translations = Euclidean(dim=n)
        self.n = n

        self.left_canonical_metric = \
            SpecialEuclideanMatrixCannonicalLeftMetric(group=self)
        self.metric = self.left_canonical_metric

    @property
    def identity(self):
        """Return the identity matrix."""
        return gs.eye(self.n + 1, self.n + 1)

    def random_point(self, n_samples=1, bound=1.):
        """Sample in SE(n) from the uniform distribution.

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
        output_shape = (
            (n_samples, self.n + 1, self.n + 1) if n_samples != 1
            else (self.n + 1, ) * 2)
        random_point = homogeneous_representation(
            random_rotation, random_translation, output_shape)
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
        translation = gs.einsum(
            '...ij,...j->...i', transposed_rot, translation)
        return homogeneous_representation(
            transposed_rot, -translation, point.shape)

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
        n = mat.shape[-1] - 1
        projected_rot = self.rotations.projection(mat[..., :n, :n])
        translation = mat[..., :n, -1]
        return homogeneous_representation(
            projected_rot, translation, mat.shape)


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

    def __init__(self, n, epsilon=0.):
        dim = n * (n + 1) // 2
        LieGroup.__init__(
            self, dim=dim, default_point_type='vector')

        self.n = n
        self.epsilon = epsilon
        self.rotations = SpecialOrthogonal(
            n=n, point_type='vector', epsilon=epsilon)
        self.translations = Euclidean(dim=n)

    def get_identity(self, point_type=None):
        """Get the identity of the group.

        Parameters
        ----------
        point_type : str, {'vector', 'matrix'}
            The point_type of the returned value.
            Optional, default: self.default_point_type

        Returns
        -------
        identity : array-like, shape={[dim], [n + 1, n + 1]}
        """
        if point_type is None:
            point_type = self.default_point_type
        identity = gs.zeros(self.dim)
        return identity
    identity = property(get_identity)

    def get_point_type_shape(self, point_type=None):
        """Get the shape of the instance given the default_point_style."""
        return self.get_identity(point_type).shape

    def belongs(self, point):
        """Evaluate if a point belongs to SE(2) or SE(3).

        Parameters
        ----------
        point : array-like, shape=[..., dimension]
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
            belongs, self.rotations.belongs(point[..., :self.rotations.dim]))
        return belongs

    def regularize(self, point):
        """Regularize a point to the default representation for SE(n).

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point to regularize.

        Returns
        -------
        point : array-like, shape=[..., 3]
            Regularized point.
        """
        rotations = self.rotations
        dim_rotations = rotations.dim

        regularized_point = point
        rot_vec = regularized_point[..., :dim_rotations]
        regularized_rot_vec = rotations.regularize(
            rot_vec)

        translation = regularized_point[..., dim_rotations:]

        return gs.concatenate(
            [regularized_rot_vec, translation], axis=-1)

    @geomstats.vectorization.decorator([
        'else', 'vector', 'else'])
    def regularize_tangent_vec_at_identity(
            self, tangent_vec, metric=None):
        """Regularize a tangent vector at the identity.

        Parameters
        ----------
        tangent_vec: array-like, shape=[..., 3]
            Tangent vector at base point.
        metric : RiemannianMetric
            Metric.
            Optional, default: None.

        Returns
        -------
        regularized_vec : array-like, shape=[..., 3]
            Regularized vector.
        """
        return self.regularize_tangent_vec(
            tangent_vec, self.identity, metric)

    @geomstats.vectorization.decorator(['else', 'vector'])
    def matrix_from_vector(self, vec):
        """Convert point in vector point-type to matrix.

        Parameters
        ----------
        vec : array-like, shape=[..., dimension]
            Vector.

        Returns
        -------
        mat : array-like, shape=[..., n+1, n+1]
            Matrix.
        """
        vec = self.regularize(vec)
        output_shape = (
            (vec.shape[0], self.n + 1, self.n + 1) if vec.ndim == 2
            else (self.n + 1, ) * 2)

        rot_vec = vec[..., :self.rotations.dim]
        trans_vec = vec[..., self.rotations.dim:]

        rot_mat = self.rotations.matrix_from_rotation_vector(rot_vec)
        return homogeneous_representation(rot_mat, trans_vec, output_shape)

    @geomstats.vectorization.decorator(
        ['else', 'vector', 'vector'])
    def compose(self, point_a, point_b):
        r"""Compose two elements of SE(2) or SE(3).

        Parameters
        ----------
        point_a : array-like, shape=[..., dimension]
            Point of the group.
        point_b : array-like, shape=[..., dimension]
            Point of the group.

        Equation
        --------
        (:math: `(R_1, t_1) \\cdot (R_2, t_2) = (R_1 R_2, R_1 t_2 + t_1)`)

        Returns
        -------
        composition : array-like, shape=[..., dimension]
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
        composition_rot_vec = rotations.rotation_vector_from_matrix(
            composition_rot_mat)

        composition_translation = gs.einsum(
            '...j,...kj->...k', translation_b, rot_mat_a) + translation_a

        composition = gs.concatenate(
            (composition_rot_vec, composition_translation), axis=-1)
        return self.regularize(composition)

    @geomstats.vectorization.decorator(['else', 'vector'])
    def inverse(self, point):
        r"""Compute the group inverse in SE(n).

        Parameters
        ----------
        point: array-like, shape=[..., dimension]
            Point.

        Returns
        -------
        inverse_point : array-like, shape=[..., dimension]
            Inverted point.

        Notes
        -----
        :math:`(R, t)^{-1} = (R^{-1}, R^{-1}.(-t))`
        """
        rotations = self.rotations
        dim_rotations = rotations.dim

        point = self.regularize(point)

        rot_vec = point[:, :dim_rotations]
        translation = point[:, dim_rotations:]

        inverse_rotation = -rot_vec

        inv_rot_mat = rotations.matrix_from_rotation_vector(
            inverse_rotation)

        inverse_translation = gs.einsum(
            'ni,nij->nj',
            -translation,
            gs.transpose(inv_rot_mat, axes=(0, 2, 1)))

        inverse_point = gs.concatenate(
            [inverse_rotation, inverse_translation], axis=-1)
        return self.regularize(inverse_point)

    @geomstats.vectorization.decorator(['else', 'vector'])
    def exp_from_identity(self, tangent_vec):
        """Compute group exponential of the tangent vector at the identity.

        Parameters
        ----------
        tangent_vec: array-like, shape=[..., 3]
            Tangent vector at base point.

        Returns
        -------
        group_exp: array-like, shape=[..., 3]
            Group exponential of the tangent vectors computed
            at the identity.
        """
        rotations = self.rotations
        dim_rotations = rotations.dim

        rot_vec = tangent_vec[..., :dim_rotations]
        rot_vec_regul = self.rotations.regularize(rot_vec)
        rot_vec_regul = gs.to_ndarray(rot_vec_regul, to_ndim=2, axis=1)

        transform = self._exp_translation_transform(rot_vec_regul)

        translation = tangent_vec[..., dim_rotations:]
        exp_translation = gs.einsum('ijk, ik -> ij', transform, translation)

        group_exp = gs.concatenate([rot_vec, exp_translation], axis=1)

        group_exp = self.regularize(group_exp)
        return group_exp

    @geomstats.vectorization.decorator(['else', 'vector'])
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

        rot_vec = point[:, :dim_rotations]
        translation = point[:, dim_rotations:]

        transform = self._log_translation_transform(rot_vec)
        log_translation = gs.einsum('ijk, ik -> ij', transform, translation)

        return gs.concatenate([rot_vec, log_translation], axis=1)

    def random_point(self, n_samples=1, bound=1., **kwargs):
        r"""Sample in SE(n) with the uniform distribution.

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
        random_point : array-like, shape=[..., dimension]
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

    def __init__(self, epsilon=0.):
        super(_SpecialEuclidean2Vectors, self).__init__(
            n=2, epsilon=epsilon)

    def regularize_tangent_vec(
            self, tangent_vec, base_point, metric=None):
        """Regularize a tangent vector at a base point.

        Parameters
        ----------
        tangent_vec: array-like, shape=[..., 3]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 3]
            Base point.
        metric : RiemannianMetric
            Metric.
            Optional, defaults to self.left_canonical_metric if None.

        Returns
        -------
        regularized_vec : array-like, shape=[..., 3]
            Regularized vector.
        """
        if metric is None:
            metric = self.left_canonical_metric

        rotations = self.rotations
        dim_rotations = rotations.dim

        rot_tangent_vec = tangent_vec[..., :dim_rotations]
        rot_base_point = base_point[..., :dim_rotations]

        rotations_vec = rotations.regularize_tangent_vec(
            tangent_vec=rot_tangent_vec,
            base_point=rot_base_point)

        return gs.concatenate(
            [rotations_vec, tangent_vec[..., dim_rotations:]], axis=-1)

    @geomstats.vectorization.decorator(['else', 'vector', 'else'])
    def jacobian_translation(self, point, left_or_right='left'):
        """Compute the Jacobian matrix resulting from translation.

        Compute the matrix of the differential of the left/right translations
        from the identity to point in SE(3).

        Parameters
        ----------
        point: array-like, shape=[..., 3]
            Point.
        left_or_right: str, {'left', 'right'}
            Whether to compute the jacobian of the left or right translation.
            Optional, default: 'left'.

        Returns
        -------
        jacobian : array-like, shape=[..., 3]
            Jacobian of the left / right translation.
        """
        if left_or_right not in ('left', 'right'):
            raise ValueError('`left_or_right` must be `left` or `right`.')

        point = self.regularize(point)

        n_points, _ = point.shape

        return gs.array([gs.eye(self.dim)] * n_points)

    def _exp_translation_transform(self, rot_vec):
        base_1 = gs.eye(2)
        base_2 = self.rotations.skew_matrix_from_vector(gs.ones(1))
        cos_coef = rot_vec * utils.taylor_exp_even_func(
            rot_vec ** 2, utils.cosc_close_0, order=3)
        sin_coef = utils.taylor_exp_even_func(
            rot_vec ** 2, utils.sinc_close_0, order=3)

        sin_term = gs.einsum('...i,...jk->...jk', sin_coef, base_1)
        cos_term = gs.einsum('...i,...jk->...jk', cos_coef, base_2)
        transform = sin_term + cos_term

        return transform

    def _log_translation_transform(self, rot_vec):
        exp_transform = self._exp_translation_transform(rot_vec)

        inv_determinant = .5 / utils.taylor_exp_even_func(
            rot_vec ** 2, utils.cosc_close_0, order=4)
        transform = gs.einsum(
            '...l, ...jk -> ...jk', inv_determinant,
            Matrices.transpose(exp_transform))

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

    def __init__(self, epsilon=0.):
        super(_SpecialEuclidean3Vectors, self).__init__(
            n=3, epsilon=epsilon)

    def regularize_tangent_vec(
            self, tangent_vec, base_point, metric=None):
        """Regularize a tangent vector at a base point.

        Parameters
        ----------
        tangent_vec: array-like, shape=[..., 3]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 3]
            Base point.
        metric : RiemannianMetric
            Metric.
            Optional, defaults to self.left_canonical_metric if None.

        Returns
        -------
        regularized_vec : array-like, shape=[..., 3]
            Regularized vector.
        """
        if metric is None:
            metric = self.left_canonical_metric

        rotations = self.rotations
        dim_rotations = rotations.dim

        rot_tangent_vec = tangent_vec[..., :dim_rotations]
        rot_base_point = base_point[..., :dim_rotations]

        metric_mat = metric.metric_mat_at_identity
        rot_metric_mat = metric_mat[:dim_rotations, :dim_rotations]
        rot_metric = InvariantMetric(group=rotations,
                                     metric_mat_at_identity=rot_metric_mat,
                                     left_or_right=metric.left_or_right)

        rotations_vec = rotations.regularize_tangent_vec(
            tangent_vec=rot_tangent_vec,
            base_point=rot_base_point,
            metric=rot_metric)

        return gs.concatenate(
            [rotations_vec, tangent_vec[..., dim_rotations:]], axis=-1)

    @geomstats.vectorization.decorator(['else', 'vector', 'else'])
    def jacobian_translation(self, point, left_or_right='left'):
        """Compute the Jacobian matrix resulting from translation.

        Compute the matrix of the differential of the left/right translations
        from the identity to point in SE(3).

        Parameters
        ----------
        point: array-like, shape=[..., 3]
            Point.
        left_or_right: str, {'left', 'right'}
            Whether to compute the jacobian of the left or right translation.
            Optional, default: 'left'.

        Returns
        -------
        jacobian : array-like, shape=[..., 3]
            Jacobian of the left / right translation.
        """
        if left_or_right not in ('left', 'right'):
            raise ValueError('`left_or_right` must be `left` or `right`.')

        rotations = self.rotations
        translations = self.translations
        dim_rotations = rotations.dim
        dim_translations = translations.dim

        point = self.regularize(point)

        n_points, _ = point.shape

        rot_vec = point[:, :dim_rotations]

        jacobian_rot = self.rotations.jacobian_translation(
            point=rot_vec, left_or_right=left_or_right)
        jacobian_rot = gs.to_ndarray(jacobian_rot, to_ndim=3)
        block_zeros_1 = gs.zeros(
            (n_points, dim_rotations, dim_translations))
        jacobian_block_line_1 = gs.concatenate(
            [jacobian_rot, block_zeros_1], axis=2)

        if left_or_right == 'left':
            rot_mat = self.rotations.matrix_from_rotation_vector(
                rot_vec)
            jacobian_trans = rot_mat
            block_zeros_2 = gs.zeros(
                (n_points, dim_translations, dim_rotations))
            jacobian_block_line_2 = gs.concatenate(
                [block_zeros_2, jacobian_trans], axis=2)

        else:
            inv_skew_mat = - self.rotations.skew_matrix_from_vector(
                rot_vec)
            eye = gs.to_ndarray(gs.eye(self.n), to_ndim=3)
            eye = gs.tile(eye, [n_points, 1, 1])
            jacobian_block_line_2 = gs.concatenate(
                [inv_skew_mat, eye], axis=2)

        jacobian = gs.concatenate(
            [jacobian_block_line_1, jacobian_block_line_2], axis=-2)
        return jacobian[0] if 1 in (len(point), point.ndim) \
            else jacobian

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

        coef_1[mask_close_to_0] = (1. / 2.
                                   - angle[mask_close_to_0] ** 2 / 24.)
        coef_2[mask_close_to_0] = (1. / 6.
                                   - angle[mask_close_to_0] ** 3 / 120.)

        # TODO (nina): Check if the discontinuity at 0 is expected.
        coef_1[mask_0] = 0
        coef_2[mask_0] = 0

        coef_1[mask_else] = (angle[mask_else] ** (-2)
                             * (1. - gs.cos(angle[mask_else])))
        coef_2[mask_else] = (angle[mask_else] ** (-2)
                             * (1. - (gs.sin(angle[mask_else])
                                      / angle[mask_else])))

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
        sq_angle = gs.sum(rot_vec ** 2, axis=-1)
        skew_mat = self.rotations.skew_matrix_from_vector(rot_vec)
        sq_skew_mat = gs.matmul(skew_mat, skew_mat)

        coef_1_ = utils.taylor_exp_even_func(
            sq_angle, utils.cosc_close_0, order=4)
        coef_2_ = utils.taylor_exp_even_func(
            sq_angle, utils.var_sinc_close_0, order=4)

        term_1 = gs.einsum('...,...ij->...ij', coef_1_, skew_mat)
        term_2 = gs.einsum('...,...ij->...ij', coef_2_, sq_skew_mat)
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
        n_samples = rot_vec.shape[0]
        angle = gs.linalg.norm(rot_vec, axis=1)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

        skew_mat = self.rotations.skew_matrix_from_vector(rot_vec)
        sq_skew_mat = gs.matmul(skew_mat, skew_mat)

        mask_close_0 = gs.isclose(angle, 0.)
        mask_close_pi = gs.isclose(angle, gs.pi)
        mask_else = ~mask_close_0 & ~mask_close_pi

        mask_close_0_float = gs.cast(mask_close_0, gs.float32)
        mask_close_pi_float = gs.cast(mask_close_pi, gs.float32)
        mask_else_float = gs.cast(mask_else, gs.float32)

        mask_0 = gs.isclose(angle, 0., atol=1e-7)
        mask_0_float = gs.cast(mask_0, gs.float32)
        angle += mask_0_float * gs.ones_like(angle)

        coef_1 = - 0.5 * gs.ones_like(angle)
        coef_2 = gs.zeros_like(angle)

        coef_2 += mask_close_0_float * (
            1. / 12. + angle ** 2 / 720.
            + angle ** 4 / 30240.
            + angle ** 6 / 1209600.)

        delta_angle = angle - gs.pi
        coef_2 += mask_close_pi_float * (
            1. / PI2
            + (PI2 - 8.) * delta_angle / (4. * PI3)
            - ((PI2 - 12.)
               * delta_angle ** 2 / (4. * PI4))
            + ((-192. + 12. * PI2 + PI4)
               * delta_angle ** 3 / (48. * PI5))
            - ((-240. + 12. * PI2 + PI4)
               * delta_angle ** 4 / (48. * PI6))
            + ((-2880. + 120. * PI2 + 10. * PI4 + PI6)
               * delta_angle ** 5 / (480. * PI7))
            - ((-3360 + 120. * PI2 + 10. * PI4 + PI6)
               * delta_angle ** 6 / (480. * PI8)))

        psi = 0.5 * angle * gs.sin(angle) / (1 - gs.cos(angle))
        coef_2 += mask_else_float * (1 - psi) / (angle ** 2)

        term_1 = gs.einsum('...i,...ij->...ij', coef_1, skew_mat)
        term_2 = gs.einsum('...i,...ij->...ij', coef_2, sq_skew_mat)
        term_id = gs.array([gs.eye(3)] * n_samples)
        transform = term_id + term_1 + term_2

        return transform


class SpecialEuclideanMatrixCannonicalLeftMetric(_InvariantMetricMatrix):
    """Class for the canonical left-invariant metric on SE(n).

    The canonical left-invariant metric is defined by endowing the tangent
    space at the identity with the Frobenius inned-product, and to define the
    metric at any point by left-translation. This results in a direct product
    metric between rotations and translations, whose geodesics are therefore
    easily computable with the matrix exponential and straight lines.

    Parameters
    ----------
    group : SpecialEuclidean
        Instance of the class SpecialEuclidean with `point_type='matrix'`.
    """

    def __init__(self, group):
        if (
                not isinstance(group, _SpecialEuclideanMatrices)
                or group.default_point_type != 'matrix'):
            raise ValueError('group must be an instance of the '
                             'SpecialEclidean class with `point_type=matrix`.')
        super(SpecialEuclideanMatrixCannonicalLeftMetric, self).__init__(
            group=group)
        self.n = group.n

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
        return Matrices.frobenius_product(tangent_vec_a, tangent_vec_b)

    def exp(self, tangent_vec, base_point=None, **kwargs):
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
        group = self.group
        if base_point is None:
            base_point = group.identity
        inf_rotation = tangent_vec[..., :self.n, :self.n]
        rotation = base_point[..., :self.n, :self.n]
        rotation_exp = GeneralLinear.exp(inf_rotation, rotation)
        translation_exp = (
            tangent_vec[..., :self.n, self.n]
            + base_point[..., :self.n, self.n])

        exp = homogeneous_representation(
            rotation_exp, translation_exp, tangent_vec.shape, 1.)
        return exp

    def log(self, point, base_point=None, **kwargs):
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
        [Zefran98]  Zefran, M., V. Kumar, and C.B. Croke.
                    “On the Generation of Smooth Three-Dimensional Rigid Body
                    Motions.” IEEE Transactions on Robotics and Automation 14,
                    no. 4 (August 1998): 576–89.
                    https://doi.org/10.1109/70.704225.
        """
        max_shape = point.shape if point.ndim == 3 else base_point.shape
        rotation_bp = base_point[..., :self.n, :self.n]
        rotation_p = point[..., :self.n, :self.n]
        rotation_log = GeneralLinear.log(rotation_p, rotation_bp)
        translation_log = (
            point[..., :self.n, self.n] - base_point[..., :self.n, self.n])

        log = homogeneous_representation(
            rotation_log, translation_log, max_shape, 0.)
        return log

    def parallel_transport(
            self, tangent_vec_a, tangent_vec_b, base_point, **kwargs):
        r"""Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector a
        along the geodesic defined by :math: `t \mapsto exp_(base_point)(t*
        tangent_vec_b)`. As the special Euclidean group endowed with its
        canonical left-invariant metric is a symmetric space, parallel
        transport is achieved by a geodesic symmetry, or equivalently, one step
         of the pole ladder scheme.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n + 1, n + 1]
            Tangent vector at base point to be transported.
        tangent_vec_b : array-like, shape=[..., n + 1, n + 1]
            Tangent vector at base point, along which the parallel transport
            is computed.
        base_point : array-like, shape=[..., n + 1, n + 1]
            Point on the hypersphere.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., n + 1, n + 1]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        """
        rot_a = tangent_vec_a[..., :self.n, :self.n]
        rot_b = tangent_vec_b[..., :self.n, :self.n]
        rot_bp = base_point[..., :self.n, :self.n]
        transported_rot = self.group.rotations.bi_invariant_metric\
            .parallel_transport(rot_a, rot_b, rot_bp)
        translation = tangent_vec_a[..., :self.n, self.n]
        max_shape = tangent_vec_a.shape
        if (tangent_vec_b.ndim == 3) and (tangent_vec_a.ndim == 2):
            translation = gs.stack([translation] * tangent_vec_b.shape[0])
            max_shape = tangent_vec_b.shape
        return homogeneous_representation(
            transported_rot, translation, max_shape, 0.)


class SpecialEuclidean(_SpecialEuclidean2Vectors,
                       _SpecialEuclidean3Vectors,
                       _SpecialEuclideanMatrices):
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

    def __new__(cls, n, point_type='matrix', epsilon=0.):
        """Instantiate a special Euclidean group.

        Select the object to instantiate depending on the point_type.
        """
        if n == 2 and point_type == 'vector':
            return _SpecialEuclidean2Vectors(epsilon)
        if n == 3 and point_type == 'vector':
            return _SpecialEuclidean3Vectors(epsilon)
        if point_type == 'vector':
            raise NotImplementedError(
                'SE(n) is only implemented in matrix representation'
                ' when n > 3.')
        return _SpecialEuclideanMatrices(n)


class SpecialEuclideanMatrixLieAlgebra(MatrixLieAlgebra):
    r"""Lie Algebra of the special Euclidean group.

    This is the tangent space at the identity. It is identified with the
    :math:`n + 1 \times n + 1` block matrices of the form:
    .. math:
                ((A, t), (0, 0))

    where A is an :math:`n \times n` skew-symmetric matrix, :math: `t` is an
    n-dimensional vector.

    Parameters
    ----------
    n : int
        Integer dimension of the underlying Euclidean space. Matrices will
        be of size: (n+1) x (n+1).
    """

    def __init__(self, n):
        dim = int(n * (n + 1) / 2)
        super(SpecialEuclideanMatrixLieAlgebra, self).__init__(dim, n)

        self.skew = SkewSymmetricMatrices(n)
        basis = homogeneous_representation(
            self.skew.basis,
            gs.zeros((self.skew.dim, n)), (self.skew.dim, n + 1, n + 1), 0.)
        basis = list(basis)

        for row in gs.arange(n):
            basis.append(gs.array_from_sparse(
                [(row, n)], [1.], (n + 1, n + 1)))
        self.basis = gs.stack(basis)

    def belongs(self, mat, atol=ATOL):
        """Evaluate if the rotation part of mat is a skew-symmetric matrix.

        Parameters
        ----------
        mat : array-like, shape=[..., n + 1, n + 1]
            Square matrix to check.
        atol : float
            Tolerance for the equality evaluation.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if rotation part of matrix is skew symmetric.
        """
        point_dim1, point_dim2 = mat.shape[-2:]
        belongs = (point_dim1 == point_dim2 == self.n + 1)

        rotation = mat[..., :self.n, :self.n]
        rot_belongs = self.skew.belongs(rotation, atol=atol)

        belongs = gs.logical_and(belongs, rot_belongs)

        last_line = mat[..., -1, :]
        all_zeros = ~ gs.any(last_line, axis=-1)

        belongs = gs.logical_and(belongs, all_zeros)
        return belongs

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
        rotation = mat[..., :self.n, :self.n]
        skew = SkewSymmetricMatrices.projection(rotation)
        return homogeneous_representation(
            skew, mat[..., :self.n, self.n], mat.shape, 0.)

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
            matrix_representation[..., :self.n, :self.n])
        translation_part = matrix_representation[..., :-1, self.n]
        return gs.concatenate([skew_part, translation_part[..., :]], axis=-1)
