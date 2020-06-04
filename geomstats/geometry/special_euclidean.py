"""The special euclidean group SE(n).

i.e. the Lie group of rigid transformations in n dimensions.
"""

import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.lie_group import LieGroup
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
TOLERANCE = 1e-8

TAYLOR_COEFFS_1_AT_0 = [+ 1. / 2., 0.,
                        - 1. / 24., 0.,
                        + 1. / 720., 0.,
                        - 1. / 40320.]

TAYLOR_COEFFS_2_AT_0 = [+ 1. / 6., 0.,
                        - 1. / 120., 0.,
                        + 1. / 5040., 0.,
                        - 1. / 362880.]


class _SpecialEuclideanMatrices(GeneralLinear, LieGroup):
    """Class for special orthogonal groups.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        super(_SpecialEuclideanMatrices, self).__init__(
            dim=int((n * (n - 1)) / 2), default_point_type='matrix', n=n + 1)
        self.rotations = SpecialOrthogonal(n=n)
        self.translations = Euclidean(dim=n)
        self.n = n

    def get_identity(self):
        """Return the identity matrix."""
        return gs.eye(self.n + 1, self.n + 1)
    identity = property(get_identity)

    def belongs(self, point):
        """Check whether point is of the form rotation, translation.

        Parameters
        ----------
        point : array-like, shape=[..., n, n].
            Point to be checked.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if point belongs to the group.
        """
        point_dim1, point_dim2 = point.shape[-2:]
        belongs = (point_dim1 == point_dim2 == self.n + 1)

        rotation = point[..., :self.n, :self.n]
        rot_belongs = self.rotations.belongs(rotation)

        belongs = gs.logical_and(belongs, rot_belongs)

        last_line_except_last_term = point[..., self.n:, :-1]
        all_but_last_zeros = ~ gs.any(
            last_line_except_last_term, axis=(-2, -1))

        belongs = gs.logical_and(belongs, all_but_last_zeros)

        last_term = point[..., self.n:, self.n:]
        belongs = gs.logical_and(
            belongs, gs.all(last_term == 1, axis=(-2, -1)))

        if point.ndim == 2:
            return gs.squeeze(belongs)
        return gs.flatten(belongs)

    def _is_in_lie_algebra(self, tangent_vec, atol=TOLERANCE):
        """Project vector rotation part onto skew-symmetric matrices."""
        point_dim1, point_dim2 = tangent_vec.shape[-2:]
        belongs = (point_dim1 == point_dim2 == self.n + 1)

        rotation = tangent_vec[..., :self.n, :self.n]
        rot_belongs = self.is_skew_symmetric(rotation, atol=atol)

        belongs = gs.logical_and(belongs, rot_belongs)

        last_line = tangent_vec[..., -1, :]
        all_zeros = ~ gs.any(last_line, axis=-1)

        belongs = gs.logical_and(belongs, all_zeros)
        return belongs

    def _to_lie_algebra(self, tangent_vec):
        """Project vector rotation part onto skew-symmetric matrices."""
        translation_mask = gs.hstack([
            gs.ones((self.n,) * 2), 2 * gs.ones((self.n, 1))])
        translation_mask = gs.concatenate(
            [translation_mask, gs.zeros((1, self.n + 1))], axis=0)
        tangent_vec = tangent_vec * gs.where(
            translation_mask != 0., gs.array(1.), gs.array(0.))
        tangent_vec = (tangent_vec - GeneralLinear.transpose(tangent_vec)) / 2.
        return tangent_vec * translation_mask

    def random_uniform(self, n_samples=1, tol=1e-6):
        """Sample in SE(n) from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        tol : unused

        Returns
        -------
        samples : array-like, shape=[..., n + 1, n + 1]
            Sample in SE(n).
        """
        random_translation = self.translations.random_uniform(n_samples)
        random_rotation = self.rotations.random_uniform(n_samples)
        random_rotation = gs.to_ndarray(random_rotation, to_ndim=3)

        random_translation = gs.to_ndarray(random_translation, to_ndim=2)
        random_translation = gs.transpose(gs.to_ndarray(
            random_translation, to_ndim=3, axis=1), (0, 2, 1))

        random_point = gs.concatenate(
            (random_rotation, random_translation), axis=2)
        last_line = gs.zeros((n_samples, 1, self.n + 1))
        random_point = gs.concatenate(
            (random_point, last_line), axis=1)
        random_point = gs.assignment(random_point, 1, (-1, -1), axis=0)
        if gs.shape(random_point)[0] == 1:
            random_point = gs.squeeze(random_point, axis=0)
        return random_point


class _SpecialEuclideanVectors(LieGroup):
    """Base Class for the special euclidean groups in 2d and 3d in vector form.

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
        n_vecs, _ = vec.shape

        rot_vec = vec[:, :self.rotations.dim]
        trans_vec = vec[:, self.rotations.dim:]

        rot_mat = self.rotations.matrix_from_rotation_vector(rot_vec)
        trans_vec = gs.reshape(trans_vec, (n_vecs, self.n, 1))
        mat = gs.concatenate((rot_mat, trans_vec), axis=2)
        last_lines = gs.array(gs.get_mask_i_float(self.n, self.n + 1))
        last_lines = gs.to_ndarray(last_lines, to_ndim=2)
        last_lines = gs.to_ndarray(last_lines, to_ndim=3)
        mat = gs.concatenate((mat, last_lines), axis=1)

        return mat

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

        composition = gs.concatenate((composition_rot_vec,
                                      composition_translation), axis=-1)
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

        group_exp = gs.concatenate(
            [rot_vec, exp_translation], axis=1)

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

        transform = self._log_translation_transform(rot_vec)

        translation = point[:, dim_rotations:]

        log_translation = gs.einsum('ijk, ik -> ij', transform, translation)

        return gs.concatenate(
            [rot_vec, log_translation], axis=1)

    def random_uniform(self, n_samples=1):
        """Sample in SE(n) with the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1

        Returns
        -------
        random_point : array-like, shape=[..., dimension]
            Sample.
        """
        random_translation = self.translations.random_uniform(n_samples)
        random_rot_vec = self.rotations.random_uniform(n_samples)
        return gs.concatenate([random_rot_vec, random_translation], axis=-1)


class _SpecialEuclidean2Vectors(_SpecialEuclideanVectors):
    """Class for the special euclidean group in 2d, SE(2).

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
        n_samples = rot_vec.shape[0]
        base_1 = gs.array([gs.eye(2)] * n_samples)
        base_2 = -gs.array(
            [self.rotations.skew_matrix_from_vector(gs.ones(1))] * n_samples)

        mask_close_0 = gs.isclose(rot_vec, 0.)
        mask_else = ~mask_close_0

        mask_close_0_float = gs.cast(mask_close_0, gs.float32)
        mask_else_float = gs.cast(mask_else, gs.float32)

        cos_coef = gs.zeros_like(rot_vec)
        sin_coef = gs.zeros_like(rot_vec)

        cos_coef += mask_close_0_float * (
            TAYLOR_COEFFS_1_AT_0[0] * rot_vec
            + TAYLOR_COEFFS_1_AT_0[1] * rot_vec ** 3
            + TAYLOR_COEFFS_1_AT_0[2] * rot_vec ** 5)
        sin_coef += mask_close_0_float * (
            1
            - TAYLOR_COEFFS_2_AT_0[0] * rot_vec ** 2
            - TAYLOR_COEFFS_2_AT_0[2] * rot_vec ** 4)

        rot_vec = rot_vec + mask_close_0_float * 1e-6
        cos_coef += mask_else_float * ((1. - gs.cos(rot_vec)) / rot_vec)
        sin_coef += mask_else_float * (gs.sin(rot_vec) / rot_vec)

        sin_term = gs.einsum('...i,...jk->...jk', sin_coef, base_1)
        cos_term = gs.einsum('...i,...jk->...jk', cos_coef, base_2)
        transform = sin_term + cos_term

        return transform

    def _log_translation_transform(self, rot_vec):
        rot_vec = gs.to_ndarray(rot_vec, to_ndim=2, axis=1)
        exp_transform = self._exp_translation_transform(rot_vec)

        if rot_vec.dtype == gs.float32:
            mask_close_0 = gs.isclose(rot_vec, 0., atol=1e-6)
        else:
            mask_close_0 = gs.isclose(rot_vec, 0.)
        mask_else = ~mask_close_0

        mask_close_0_float = gs.cast(mask_close_0, gs.float32)
        mask_else_float = gs.cast(mask_else, gs.float32)

        inv_determinant = gs.zeros_like(rot_vec)

        inv_determinant += 0.5 * mask_close_0_float / (
            TAYLOR_COEFFS_1_AT_0[0]
            + TAYLOR_COEFFS_1_AT_0[1] * rot_vec ** 2
            + TAYLOR_COEFFS_1_AT_0[2] * rot_vec ** 6)

        rot_vec = rot_vec + 1e-3 * mask_close_0_float
        inv_determinant += mask_else_float * (
            rot_vec ** 2 / (2 * (1 - gs.cos(rot_vec))))
        transform = gs.einsum(
            'il, ijk -> ijk', inv_determinant,
            gs.transpose(exp_transform, axes=[0, 2, 1]))

        return transform


class _SpecialEuclidean3Vectors(_SpecialEuclideanVectors):
    """Class for the special euclidean group in 3d, SE(3).

    i.e. the Lie group of rigid transformations. Elements of SE(3) can either
    be represented as vectors (in 3d) or as matrices in general. The matrix
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

        metric_mat = metric.inner_product_mat_at_identity
        rot_metric_mat = metric_mat[:dim_rotations, :dim_rotations]
        rot_metric = InvariantMetric(
            group=rotations,
            inner_product_mat_at_identity=rot_metric_mat,
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
        return jacobian[0] if (len(point) == 1 or point.ndim == 1) \
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
        n_samples = rot_vec.shape[0]

        angle = gs.linalg.norm(rot_vec, axis=-1)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

        skew_mat = self.rotations.skew_matrix_from_vector(rot_vec)
        sq_skew_mat = gs.matmul(skew_mat, skew_mat)

        mask_0 = gs.equal(angle, 0.)
        mask_close_0 = gs.isclose(angle, 0.) & ~mask_0
        mask_else = ~mask_0 & ~mask_close_0

        mask_0_float = gs.cast(mask_0, gs.float32)
        mask_close_0_float = gs.cast(mask_close_0, gs.float32)
        mask_else_float = gs.cast(mask_else, gs.float32)

        coef_1 = gs.zeros_like(angle)
        coef_2 = gs.zeros_like(angle)

        coef_1 += mask_0_float * 1. / 2. * gs.ones_like(angle)
        coef_2 += mask_0_float * 1. / 6. * gs.ones_like(angle)

        coef_1 += mask_close_0_float * (
            TAYLOR_COEFFS_1_AT_0[0]
            + TAYLOR_COEFFS_1_AT_0[2] * angle ** 2
            + TAYLOR_COEFFS_1_AT_0[4] * angle ** 4
            + TAYLOR_COEFFS_1_AT_0[6] * angle ** 6)
        coef_2 += mask_close_0_float * (
            TAYLOR_COEFFS_2_AT_0[0]
            + TAYLOR_COEFFS_2_AT_0[2] * angle ** 2
            + TAYLOR_COEFFS_2_AT_0[4] * angle ** 4
            + TAYLOR_COEFFS_2_AT_0[6] * angle ** 6)

        angle += mask_0_float * 1.

        coef_1 += mask_else_float * ((1. - gs.cos(angle)) / angle ** 2)
        coef_2 += mask_else_float * ((angle - gs.sin(angle)) / angle ** 3)

        term_1 = gs.einsum('...i,...ij->...ij', coef_1, skew_mat)
        term_2 = gs.einsum('...i,...ij->...ij', coef_2, sq_skew_mat)
        term_id = gs.array([gs.eye(3)] * n_samples)
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

        mask_0 = gs.isclose(angle, 0., atol=1e-6)
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


class SpecialEuclidean(_SpecialEuclidean2Vectors,
                       _SpecialEuclidean3Vectors,
                       _SpecialEuclideanMatrices):
    r"""Class for the special euclidean groups.

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
        """Instantiate a special euclidean group.

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
