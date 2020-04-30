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
        n_samples : int, optional (1)
            Number of samples.
        tol :  unused

        Returns
        -------
        samples : array-like, shape=[..., n + 1, n + 1]
            Points sampled on the SE(n).
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


class _SpecialEuclidean3Vectors(LieGroup):
    """Class for the special euclidean group in 3d, SE(3).

    i.e. the Lie group of rigid transformations. Elements of SE(3) can either
    be represented as vectors (in 3d) or as matrices in general. The matrix
    representation corresponds to homogeneous coordinates.This class is
    specific to the vector representation of rotations. For the matrix
    representation use the SpecialEuclidean class and set `n=3`.

    Parameter
    ---------
    epsilon : float, optional (defaults to 0)
        Precision to use for calculations involving potential
        division by 0 in rotations.
    """

    def __init__(self, epsilon=0.):
        super(_SpecialEuclidean3Vectors, self).__init__(
            dim=6, default_point_type='vector')

        self.n = 3
        self.epsilon = epsilon
        self.rotations = SpecialOrthogonal(
            n=3, point_type='vector', epsilon=epsilon)
        self.translations = Euclidean(dim=3)

    def get_identity(self, point_type=None):
        """Get the identity of the group.

        Parameters
        ----------
        point_type : str, {'vector', 'matrix'}, optional
            The point_type of the returned value.
            default: self.default_point_type

        Returns
        -------
        identity : array-like, shape={[dim], [n + 1, n + 1]}
        """
        if point_type is None:
            point_type = self.default_point_type
        identity = gs.zeros(self.dim)
        if point_type == 'matrix':
            identity = gs.eye(self.n + 1)
        return identity
    identity = property(get_identity)

    def get_point_type_shape(self, point_type=None):
        """Get the shape of the instance given the default_point_style."""
        return self.get_identity(point_type).shape

    def belongs(self, point):
        """Evaluate if a point belongs to SE(3).

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            The point of which to check whether it belongs to SE(3).

        Returns
        -------
        belongs : array-like, shape=[..., 1]
            Boolean indicating whether point belongs to SE(3).
        """
        point_dim = point.shape[-1]
        point_ndim = point.ndim
        belongs = gs.logical_and(point_dim == self.dim, point_ndim < 3)

        belongs = gs.logical_and(
            belongs, self.rotations.belongs(point[..., :self.n]))
        return belongs

    def regularize(self, point):
        """Regularize a point to the default representation for SE(n).

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            The point to regularize.

        Returns
        -------
        point : array-like, shape=[..., 3]
        """
        rotations = self.rotations
        dim_rotations = rotations.dim

        rot_vec = point[..., :dim_rotations]
        regularized_rot_vec = rotations.regularize(
            rot_vec)

        translation = point[..., dim_rotations:]

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
        metric : RiemannianMetric, optional

        Returns
        -------
        regularized_vec : the regularized tangent vector
        """
        return self.regularize_tangent_vec(
            tangent_vec, self.identity, metric)

    def regularize_tangent_vec(
            self, tangent_vec, base_point, metric=None):
        """Regularize a tangent vector at a base point.

        Parameters
        ----------
        tangent_vec: array-like, shape=[..., 3]
        base_point : array-like, shape=[..., 3]
        metric : RiemannianMetric, optional
            default: self.left_canonical_metric

        Returns
        -------
        regularized_vec : the regularized tangent vector
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

    @geomstats.vectorization.decorator(['else', 'vector'])
    def matrix_from_vector(self, vec):
        """Convert point in vector point-type to matrix.

        Parameters
        ----------
        vec: array-like, shape=[..., 3]

        Returns
        -------
        mat: array-like, shape=[..., n+1, n+1]
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
        r"""Compose two elements of SE(3).

        Parameters
        ----------
        point_a : array-like, shape=[..., 3]
            Point of the group.
        point_b : array-like, shape=[..., 3]
            Point of the group.

        Equation
        ---------
        (:math: `(R_1, t_1) \\cdot (R_2, t_2) = (R_1 R_2, R_1 t_2 + t_1)`)

        Returns
        -------
        composition :
            The composition of point_a and point_b.

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
        point: array-like, shape=[..., 3]

        Returns
        -------
        inverse_point : array-like, shape=[..., 3]
            The inverted point.

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

    @geomstats.vectorization.decorator(['else', 'vector', 'else'])
    def jacobian_translation(self, point, left_or_right='left'):
        """Compute the Jacobian matrix resulting from translation.

        Compute the matrix of the differential of the left/right translations
        from the identity to point in SE(3).

        Parameters
        ----------
        point: array-like, shape=[..., 3]
        left_or_right: str, {'left', 'right'}, optional
            Whether to compute the jacobian of the left or right translation.

        Returns
        -------
        jacobian : array-like, shape=[..., 3]
            The jacobian of the left / right translation.
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
            point=rot_vec,
            left_or_right=left_or_right)
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

        return gs.concatenate(
            [jacobian_block_line_1, jacobian_block_line_2], axis=1)

    @geomstats.vectorization.decorator(['else', 'vector'])
    def exp_from_identity(self, tangent_vec):
        """Compute group exponential of the tangent vector at the identity.

        Parameters
        ----------
        tangent_vec: array-like, shape=[..., 3]

        Returns
        -------
        group_exp: array-like, shape=[..., 3]
            The group exponential of the tangent vectors calculated
            at the identity.
        """
        rotations = self.rotations
        dim_rotations = rotations.dim

        rot_vec = tangent_vec[..., :dim_rotations]
        rot_vec = self.rotations.regularize(rot_vec)
        translation = tangent_vec[..., dim_rotations:]

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

        angle += mask_0_float * gs.ones_like(angle)

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

        coef_1 += mask_else_float * ((1. - gs.cos(angle)) / angle ** 2)
        coef_2 += mask_else_float * ((angle - gs.sin(angle)) / angle ** 3)

        n_tangent_vecs, _ = tangent_vec.shape
        exp_translation = gs.zeros((n_tangent_vecs, self.n))
        for i in range(n_tangent_vecs):
            translation_i = translation[i]
            term_1_i = coef_1[i] * gs.dot(translation_i,
                                          gs.transpose(skew_mat[i]))
            term_2_i = coef_2[i] * gs.dot(translation_i,
                                          gs.transpose(sq_skew_mat[i]))
            mask_i_float = gs.get_mask_i_float(i, n_tangent_vecs)
            exp_translation += gs.outer(
                mask_i_float, translation_i + term_1_i + term_2_i)

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

        Returns
        -------
        group_log: array-like, shape=[..., 3]
            the group logarithm in the Lie algbra
        """
        point = self.regularize(point)

        rotations = self.rotations
        dim_rotations = rotations.dim

        rot_vec = point[:, :dim_rotations]
        angle = gs.linalg.norm(rot_vec, axis=1)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

        translation = point[:, dim_rotations:]

        skew_rot_vec = rotations.skew_matrix_from_vector(rot_vec)
        sq_skew_rot_vec = gs.matmul(skew_rot_vec, skew_rot_vec)

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

        n_points, _ = point.shape
        log_translation = gs.zeros((n_points, self.n))
        for i in range(n_points):
            translation_i = translation[i]
            term_1_i = coef_1[i] * gs.dot(translation_i,
                                          gs.transpose(skew_rot_vec[i]))
            term_2_i = coef_2[i] * gs.dot(translation_i,
                                          gs.transpose(sq_skew_rot_vec[i]))
            mask_i_float = gs.get_mask_i_float(i, n_points)
            log_translation += gs.outer(
                mask_i_float, translation_i + term_1_i + term_2_i)

        return gs.concatenate(
            [rot_vec, log_translation], axis=1)

    def random_uniform(self, n_samples=1):
        """Sample in SE(3) with the uniform distribution.

        Parameters
        ----------
        n_samples : int, optional
            default : 1

        Returns
        -------
        random_point : array-like, shape=[..., 3]
            An array of random elements in SE(3) having the given.
        """
        random_translation = self.translations.random_uniform(n_samples)
        random_rot_vec = self.rotations.random_uniform(n_samples)
        return gs.concatenate([random_rot_vec, random_translation], axis=-1)

    def _exponential_matrix(self, rot_vec):
        """Compute exponential of rotation matrix represented by rot_vec.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]

        Returns
        -------
        exponential_mat : The matrix exponential of rot_vec
        """
        # TODO (nguigs): find usecase for this method
        rot_vec = self.rotations.regularize(rot_vec)
        n_rot_vecs, _ = rot_vec.shape

        angle = gs.linalg.norm(rot_vec, axis=1)
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


class SpecialEuclidean(_SpecialEuclidean3Vectors, _SpecialEuclideanMatrices):
    r"""Class for the special euclidean groups.

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

    def __new__(cls, n, point_type='matrix', epsilon=0.):
        """Instantiate a special euclidean group.

        Select the object to instantiate depending on the point_type.
        """
        if n == 3 and point_type == 'vector':
            return _SpecialEuclidean3Vectors(epsilon)
        if n != 3 and point_type == 'vector':
            raise NotImplementedError(
                'SE(n) is only implemented in matrix representation'
                ' when n != 3.')
        return _SpecialEuclideanMatrices(n)
