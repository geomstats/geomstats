"""
The special euclidean group SE(n),
i.e. the Lie group of rigid transformations in n dimensions.
"""

import geomstats.backend as gs

from geomstats.euclidean_space import EuclideanSpace
from geomstats.invariant_metric import InvariantMetric
from geomstats.lie_group import LieGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

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


class SpecialEuclideanGroup(LieGroup):
    """
    Class for the special euclidean group SE(n),
    i.e. the Lie group of rigid transformations.
    """

    def __init__(self, n, point_type=None):
        assert isinstance(n, int) and n > 1

        self.n = n
        self.dimension = int((n * (n - 1)) / 2 + n)

        self.default_point_type = point_type
        if point_type is None:
            self.default_point_type = 'vector' if n == 3 else 'matrix'

        super(SpecialEuclideanGroup, self).__init__(
                          dimension=self.dimension)

        # TODO(nina): keep the names rotations and translations here?
        self.rotations = SpecialOrthogonalGroup(n=n)
        self.translations = EuclideanSpace(dimension=n)

    def get_identity(self, point_type=None):
        """
        Get the identity of the group,
        as a vector if point_type == 'vector',
        as a matrix if point_type == 'matrix'.
        """
        if point_type is None:
            point_type = self.default_point_type

        identity = gs.zeros(self.dimension)
        if self.default_point_type == 'matrix':
            identity = gs.eye(self.n)
        return identity
    identity = property(get_identity)

    def belongs(self, point, point_type=None):
        """
        Evaluate if a point belongs to SE(n).
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            point = gs.to_ndarray(point, to_ndim=2)
            _, point_dim = point.shape
            return point_dim == self.dimension
        elif point_type == 'matrix':
            point = gs.to_ndarray(point, to_ndim=3)
            raise NotImplementedError()

    def regularize(self, point, point_type=None):
        """
        Regularize a point to the canonical representation
        chosen for SE(n).
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            point = gs.to_ndarray(point, to_ndim=2)
            assert self.belongs(point, point_type=point_type)

            rotations = self.rotations
            dim_rotations = rotations.dimension

            regularized_point = gs.zeros_like(point)
            rot_vec = point[:, :dim_rotations]
            regularized_point[:, :dim_rotations] = rotations.regularize(
                rot_vec, point_type=point_type)
            regularized_point[:, dim_rotations:] = point[:, dim_rotations:]

        elif point_type == 'matrix':
            point = gs.to_ndarray(point, to_ndim=3)
            # TODO(nina): regularization for matrices?
            regularized_point = gs.copy(point)

        return regularized_point

    def regularize_tangent_vec_at_identity(
            self, tangent_vec, metric=None, point_type=None):
        if point_type is None:
            point_type = self.default_point_type

        return self.regularize_tangent_vec(
                tangent_vec, self.identity, metric, point_type=point_type)

    def regularize_tangent_vec(
            self, tangent_vec, base_point, metric=None, point_type=None):
        if point_type is None:
            point_type = self.default_point_type

        if metric is None:
            metric = self.left_canonical_metric

        if point_type == 'vector':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)

            rotations = self.rotations
            dim_rotations = rotations.dimension

            rot_tangent_vec = tangent_vec[:, :dim_rotations]
            rot_base_point = base_point[:, :dim_rotations]

            metric_mat = metric.inner_product_mat_at_identity
            rot_metric_mat = metric_mat[:, :dim_rotations, :dim_rotations]
            rot_metric = InvariantMetric(
                               group=rotations,
                               inner_product_mat_at_identity=rot_metric_mat,
                               left_or_right=metric.left_or_right)

            regularized_vec = gs.zeros_like(tangent_vec)
            rotations_vec = rotations.regularize_tangent_vec(
                                             tangent_vec=rot_tangent_vec,
                                             base_point=rot_base_point,
                                             metric=rot_metric,
                                             point_type=point_type)

            regularized_vec[:, :dim_rotations] = rotations_vec
            regularized_vec[:, dim_rotations:] = tangent_vec[:, dim_rotations:]

        elif point_type == 'matrix':
                # TODO(nina): regularization in terms
                # of skew-symmetric matrices?
                regularized_vec = tangent_vec

        return regularized_vec

    def compose(self, point_1, point_2, point_type=None):
        """
        Compose two elements of SE(n).

        Formula:
        point_1 . point_2 = [R1 * R2, (R1 * t2) + t1]
        where:
        R1, R2 are rotation matrices,
        t1, t2 are translation vectors.
        """
        if point_type is None:
            point_type = self.default_point_type

        rotations = self.rotations
        dim_rotations = rotations.dimension

        point_1 = self.regularize(point_1, point_type=point_type)
        point_2 = self.regularize(point_2, point_type=point_type)

        if point_type == 'vector':
            n_points_1, _ = point_1.shape
            n_points_2, _ = point_2.shape

            assert (point_1.shape == point_2.shape
                    or n_points_1 == 1
                    or n_points_2 == 1)

            rot_vec_1 = point_1[:, :dim_rotations]
            rot_mat_1 = rotations.matrix_from_rotation_vector(rot_vec_1)
            rot_mat_1 = rotations.projection(rot_mat_1)

            rot_vec_2 = point_2[:, :dim_rotations]
            rot_mat_2 = rotations.matrix_from_rotation_vector(rot_vec_2)
            rot_mat_2 = rotations.projection(rot_mat_2)

            translation_1 = point_1[:, dim_rotations:]
            translation_2 = point_2[:, dim_rotations:]

            n_compositions = gs.maximum(n_points_1, n_points_2)
            composition_rot_mat = gs.matmul(rot_mat_1, rot_mat_2)
            composition_rot_vec = rotations.rotation_vector_from_matrix(
                                                          composition_rot_mat)
            composition_translation = gs.zeros((n_compositions, self.n))
            for i in range(n_compositions):
                translation_1_i = (translation_1[0] if n_points_1 == 1
                                   else translation_1[i])
                rot_mat_1_i = (rot_mat_1[0] if n_points_1 == 1
                               else rot_mat_1[i])
                translation_2_i = (translation_2[0] if n_points_2 == 1
                                   else translation_2[i])
                composition_translation[i] = (gs.dot(translation_2_i,
                                                     gs.transpose(rot_mat_1_i))
                                              + translation_1_i)

            composition = gs.zeros((n_compositions, self.dimension))
            composition[:, :dim_rotations] = composition_rot_vec
            composition[:, dim_rotations:] = composition_translation

        elif point_type == 'matrix':
            raise NotImplementedError()

        composition = self.regularize(composition, point_type=point_type)
        return composition

    def inverse(self, point, point_type=None):
        """
        Compute the group inverse in SE(n).

        Formula:
        (R, t)^{-1} = (R^{-1}, R^{-1}.(-t))
        """
        if point_type is None:
            point_type = self.default_point_type

        rotations = self.rotations
        dim_rotations = rotations.dimension

        point = self.regularize(point)

        if point_type == 'vector':
            n_points, _ = point.shape

            rot_vec = point[:, :dim_rotations]
            translation = point[:, dim_rotations:]

            inverse_point = gs.zeros_like(point)
            inverse_rotation = -rot_vec

            inv_rot_mat = rotations.matrix_from_rotation_vector(
                inverse_rotation)

            inverse_translation = gs.zeros((n_points, self.n))
            for i in range(n_points):
                inverse_translation[i] = gs.dot(-translation[i],
                                                gs.transpose(inv_rot_mat[i]))

            inverse_point[:, :dim_rotations] = inverse_rotation
            inverse_point[:, dim_rotations:] = inverse_translation

        elif point_type == 'matrix':
            raise NotImplementedError()

        inverse_point = self.regularize(inverse_point, point_type=point_type)
        return inverse_point

    def jacobian_translation(
            self, point, left_or_right='left', point_type=None):
        """
        Compute the jacobian matrix of the differential
        of the left/right translations from the identity to point in SE(n).
        """
        if point_type is None:
            point_type = self.default_point_type

        assert self.belongs(point, point_type=point_type)
        assert left_or_right in ('left', 'right')

        dim = self.dimension
        rotations = self.rotations
        dim_rotations = rotations.dimension

        point = self.regularize(point, point_type=point_type)

        if point_type == 'vector':
            n_points, _ = point.shape

            rot_vec = point[:, :dim_rotations]

            jacobian = gs.zeros((n_points,) + (dim,) * 2)
            jacobian_rot = self.rotations.jacobian_translation(
                                          point=rot_vec,
                                          left_or_right=left_or_right,
                                          point_type=point_type)
            jacobian[:, :dim_rotations, :dim_rotations] = jacobian_rot

            if left_or_right == 'left':
                rot_mat = self.rotations.matrix_from_rotation_vector(
                        rot_vec)
                jacobian_trans = rot_mat

                jacobian[:, dim_rotations:, dim_rotations:] = jacobian_trans

            else:
                inv_skew_mat = - self.rotations.skew_matrix_from_vector(
                    rot_vec)

                jacobian[:, dim_rotations:, :dim_rotations] = inv_skew_mat
                jacobian[:, dim_rotations:, dim_rotations:] = gs.eye(self.n)

            assert jacobian.ndim == 3

        elif point_type == 'matrix':
            raise NotImplementedError()

        return jacobian

    def group_exp_from_identity(self, tangent_vec, point_type=None):
        """
        Compute the group exponential of the tangent vector at the identity.
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)

            rotations = self.rotations
            dim_rotations = rotations.dimension

            rot_vec = tangent_vec[:, :dim_rotations]
            rot_vec = self.rotations.regularize(rot_vec, point_type=point_type)
            translation = tangent_vec[:, dim_rotations:]

            angle = gs.linalg.norm(rot_vec, axis=1)
            angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

            mask_close_pi = gs.isclose(angle, gs.pi)
            mask_close_pi = gs.squeeze(mask_close_pi, axis=1)
            rot_vec[mask_close_pi] = rotations.regularize(
                                           rot_vec[mask_close_pi],
                                           point_type=point_type)

            skew_mat = self.rotations.skew_matrix_from_vector(rot_vec)
            sq_skew_mat = gs.matmul(skew_mat, skew_mat)

            mask_0 = gs.equal(angle, 0)
            mask_close_0 = gs.isclose(angle, 0) & ~mask_0

            mask_0 = gs.squeeze(mask_0, axis=1)
            mask_close_0 = gs.squeeze(mask_close_0, axis=1)

            mask_else = ~mask_0 & ~mask_close_0

            coef_1 = gs.zeros_like(angle)
            coef_2 = gs.zeros_like(angle)

            coef_1[mask_0] = 1. / 2.
            coef_2[mask_0] = 1. / 6.

            coef_1[mask_close_0] = (
                TAYLOR_COEFFS_1_AT_0[0]
                + TAYLOR_COEFFS_1_AT_0[2] * angle[mask_close_0] ** 2
                + TAYLOR_COEFFS_1_AT_0[4] * angle[mask_close_0] ** 4
                + TAYLOR_COEFFS_1_AT_0[6] * angle[mask_close_0] ** 6)
            coef_2[mask_close_0] = (
                TAYLOR_COEFFS_2_AT_0[0]
                + TAYLOR_COEFFS_2_AT_0[2] * angle[mask_close_0] ** 2
                + TAYLOR_COEFFS_2_AT_0[4] * angle[mask_close_0] ** 4
                + TAYLOR_COEFFS_2_AT_0[6] * angle[mask_close_0] ** 6)

            coef_1[mask_else] = ((1. - gs.cos(angle[mask_else]))
                                 / angle[mask_else] ** 2)
            coef_2[mask_else] = ((angle[mask_else] - gs.sin(angle[mask_else]))
                                 / angle[mask_else] ** 3)

            n_tangent_vecs, _ = tangent_vec.shape
            group_exp_translation = gs.zeros((n_tangent_vecs, self.n))
            for i in range(n_tangent_vecs):
                translation_i = translation[i]
                term_1_i = coef_1[i] * gs.dot(translation_i,
                                              gs.transpose(skew_mat[i]))
                term_2_i = coef_2[i] * gs.dot(translation_i,
                                              gs.transpose(sq_skew_mat[i]))

                group_exp_translation[i] = translation_i + term_1_i + term_2_i

            group_exp = gs.zeros_like(tangent_vec)
            group_exp[:, :dim_rotations] = rot_vec
            group_exp[:, dim_rotations:] = group_exp_translation

            group_exp = self.regularize(group_exp, point_type=point_type)
            return group_exp
        elif point_type == 'matrix':
            raise NotImplementedError()

    def group_log_from_identity(self, point, point_type=None):
        """
        Compute the group logarithm of the point at the identity.
        """
        if point_type is None:
            point_type = self.default_point_type

        assert self.belongs(point, point_type=point_type)
        point = self.regularize(point, point_type=point_type)

        rotations = self.rotations
        dim_rotations = rotations.dimension

        if point_type == 'vector':
            rot_vec = point[:, :dim_rotations]
            angle = gs.linalg.norm(rot_vec, axis=1)
            angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

            translation = point[:, dim_rotations:]

            group_log = gs.zeros_like(point)
            group_log[:, :dim_rotations] = rot_vec
            skew_rot_vec = rotations.skew_matrix_from_vector(rot_vec)
            sq_skew_rot_vec = gs.matmul(skew_rot_vec, skew_rot_vec)

            mask_close_0 = gs.isclose(angle, 0)
            mask_close_0 = gs.squeeze(mask_close_0, axis=1)

            mask_close_pi = gs.isclose(angle, gs.pi)
            mask_close_pi = gs.squeeze(mask_close_pi, axis=1)

            mask_else = ~mask_close_0 & ~mask_close_pi

            coef_1 = - 0.5 * gs.ones_like(angle)
            coef_2 = gs.zeros_like(angle)

            coef_2[mask_close_0] = (1. / 12. + angle[mask_close_0] ** 2 / 720.
                                    + angle[mask_close_0] ** 4 / 30240.
                                    + angle[mask_close_0] ** 6 / 1209600.)

            delta_angle = angle[mask_close_pi] - gs.pi
            coef_2[mask_close_pi] = (1. / PI2
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

            psi = (0.5 * angle[mask_else]
                   * gs.sin(angle[mask_else]) / (1 - gs.cos(angle[mask_else])))
            coef_2[mask_else] = (1 - psi) / (angle[mask_else] ** 2)

            n_points, _ = point.shape
            group_log_translation = gs.zeros((n_points, self.n))
            for i in range(n_points):
                translation_i = translation[i]
                term_1_i = coef_1[i] * gs.dot(translation_i,
                                              gs.transpose(skew_rot_vec[i]))
                term_2_i = coef_2[i] * gs.dot(translation_i,
                                              gs.transpose(sq_skew_rot_vec[i]))
                group_log_translation[i] = translation_i + term_1_i + term_2_i

            group_log[:, dim_rotations:] = group_log_translation

            assert group_log.ndim == 2

        elif point_type == 'matrix':
            raise NotImplementedError()

        return group_log

    def random_uniform(self, n_samples=1, point_type=None):
        """
        Sample in SE(n) with the uniform distribution.
        """
        if point_type is None:
            point_type = self.default_point_type

        random_rot_vec = self.rotations.random_uniform(
            n_samples, point_type=point_type)
        random_translation = self.translations.random_uniform(n_samples)

        if point_type == 'vector':
            random_transfo = gs.concatenate(
                [random_rot_vec, random_translation],
                axis=1)

        elif point_type == 'matrix':
            raise NotImplementedError()

        random_transfo = self.regularize(random_transfo, point_type=point_type)
        return random_transfo

    def exponential_matrix(self, rot_vec):
        """
        Compute the exponential of the rotation matrix represented by rot_vec.
        """

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

        # TODO(nina): check if the discountinuity as 0 is expected.
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
        assert exponential_mat.ndim == 3

        return exponential_mat

    def group_exponential_barycenter(
            self, points, weights=None, point_type=None):
        """
        Compute the group exponential barycenter in SE(n).
        """
        if point_type is None:
            point_type = self.default_point_type

        n_points = points.shape[0]
        assert n_points > 0

        if weights is None:
            weights = gs.ones((n_points, 1))

        weights = gs.to_ndarray(weights, to_ndim=2, axis=1)
        n_weights, _ = weights.shape
        assert n_points == n_weights

        dim = self.dimension
        rotations = self.rotations
        dim_rotations = rotations.dimension

        if point_type == 'vector':
            rotation_vectors = points[:, :dim_rotations]
            translations = points[:, dim_rotations:dim]
            assert rotation_vectors.shape == (n_points, dim_rotations)
            assert translations.shape == (n_points, self.n)

            mean_rotation = rotations.group_exponential_barycenter(
                                                    points=rotation_vectors,
                                                    weights=weights)
            mean_rotation_mat = rotations.matrix_from_rotation_vector(
                        mean_rotation)

            matrix = gs.zeros((1,) + (self.n,) * 2)
            translation_aux = gs.zeros((1, self.n))

            inv_rot_mats = rotations.matrix_from_rotation_vector(
                    -rotation_vectors)
            # TODO(nina): this is the same mat multiplied several times
            matrix_aux = gs.matmul(mean_rotation_mat, inv_rot_mats)
            assert matrix_aux.shape == (n_points,) + (dim_rotations,) * 2

            vec_aux = rotations.rotation_vector_from_matrix(matrix_aux)
            matrix_aux = self.exponential_matrix(vec_aux)
            matrix_aux = gs.linalg.inv(matrix_aux)

            for i in range(n_points):
                matrix += weights[i] * matrix_aux[i]
                translation_aux += weights[i] * gs.dot(gs.matmul(
                                                            matrix_aux[i],
                                                            inv_rot_mats[i]),
                                                       translations[i])

            mean_translation = gs.dot(translation_aux,
                                      gs.transpose(gs.linalg.inv(matrix),
                                                   axes=(0, 2, 1)))

            exp_bar = gs.zeros((1, dim))
            exp_bar[0, :dim_rotations] = mean_rotation
            exp_bar[0, dim_rotations:dim] = mean_translation

        elif point_type == 'matrix':
            vector_points = self.rotation_vector_from_matrix(points)
            vector_exp_bar = self.group_exponential_barycenter(
                vector_points, weights, point_type='vector')
            exp_bar = self.matrix_from_rotation_vector(vector_exp_bar)
        return exp_bar
