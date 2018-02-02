"""Computations on the Lie group of rigid transformations."""

import numpy as np

import geomstats.special_orthogonal_group as so_group

from geomstats.euclidean_space import EuclideanSpace
from geomstats.lie_group import LieGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


class SpecialEuclideanGroup(LieGroup):

    def __init__(self, n):
        assert n > 1

        if n is not 3:
            raise NotImplementedError('Only SE(3) is implemented.')

        self.n = n
        self.dimension = int((n * (n - 1)) / 2 + n)
        super(SpecialEuclideanGroup, self).__init__(
                          dimension=self.dimension,
                          identity=np.zeros(self.dimension))
        self.rotations = SpecialOrthogonalGroup(n=n)
        self.translations = EuclideanSpace(dimension=n)

    def belongs(self, transfo):
        """
        Check that the transformation belongs to
        the special euclidean group.
        """
        if transfo.ndim == 1:
            transfo = np.expand_dims(transfo, axis=0)
        assert transfo.ndim == 2

        return transfo.shape[1] == self.dimension

    def regularize(self, transfo):
        """
        Regularize an element of the group SE(3),
        by extracting the rotation vector r from the input [r t]
        and using self.rotations.regularize.

        :param transfo: 6d vector, element in SE(3) represented as [r t].
        :returns self.regularized_transfo: 6d vector, element in SE(3)
        with self.regularized rotation.
        """
        assert self.belongs(transfo)
        if transfo.ndim == 1:
            transfo = np.expand_dims(transfo, axis=0)
        assert transfo.ndim == 2

        regularized_transfo = np.zeros_like(transfo)
        rot_vec = transfo[:, 0:3]
        regularized_transfo[:, :3] = self.rotations.regularize(rot_vec)
        regularized_transfo[:, 3:6] = transfo[:, 3:6]

        return regularized_transfo

    def compose(self, transfo_1, transfo_2):
        """
        Compose two elements of group SE(3).

        Formula:
        transfo_1 . transfo_2 = [R1 * R2, (R1 * t2) + t1]
        where:
        R1, R2 are rotation matrices,
        t1, t2 are translation vectors.

        :param transfo_1, transfo_2: 6d vectors elements of SE(3)
        :returns prod_transfo: composition of transfo_1 and transfo_2
        """
        transfo_1 = self.regularize(transfo_1)
        transfo_2 = self.regularize(transfo_2)

        rot_vec_1 = transfo_1[:, 0:3]
        rot_mat_1 = self.rotations.matrix_from_rotation_vector(rot_vec_1)

        rot_mat_1 = so_group.closest_rotation_matrix(rot_mat_1)

        rot_vec_2 = transfo_2[:, 0:3]
        rot_mat_2 = self.rotations.matrix_from_rotation_vector(rot_vec_2)
        rot_mat_2 = so_group.closest_rotation_matrix(rot_mat_2)

        translation_1 = transfo_1[:, 3:6]
        translation_2 = transfo_2[:, 3:6]

        prod_transfo = np.zeros((rot_mat_1.shape[0], 6))
        prod_rot_mat = np.matmul(rot_mat_1, rot_mat_2)

        prod_transfo[:, :3] = self.rotations.rotation_vector_from_matrix(
                                                                  prod_rot_mat)
        prod_transfo[:, 3:6] = (np.dot(translation_2,
                                       np.transpose(rot_mat_1, axes=(0, 2, 1)))
                                + translation_1)

        prod_transfo = self.regularize(prod_transfo)
        return prod_transfo

    def inverse(self, transfo):
        """
        Compute the group inverse in SE(3).

        Formula:
        (R, t)^{-1} = (R^{-1}, R^{-1}.(-t))

        :param transfo: 6d vector element in SE(3)
        :returns inverse_transfo: 6d vector inverse of transfo
        """
        transfo = self.regularize(transfo)

        rot_vec = transfo[:, 0:3]
        translation = transfo[:, 3:6]

        inverse_transfo = np.zeros_like(transfo)
        inverse_transfo[:, 0:3] = -rot_vec
        rot_mat = self.rotations.matrix_from_rotation_vector(-rot_vec)
        inverse_transfo[:, 3:6] = np.dot(-translation,
                                         np.transpose(rot_mat, axes=(0, 2, 1)))

        inverse_transfo = self.regularize(inverse_transfo)
        return inverse_transfo

    def jacobian_translation(self, point, left_or_right='left'):
        """
        Compute the jacobian matrix of the differential
        of the left/right translations
        from the identity to point in the Lie group SE(3).

        :param point: 6D vector element of SE(3)
        :returns jacobian: 6x6 matrix
        """
        assert self.belongs(point)
        assert left_or_right in ('left', 'right')

        point = self.regularize(point)
        rot_vec = point[:, 0:3]

        jacobian = np.zeros((point.shape[0], 6, 6))

        if left_or_right == 'left':
            jacobian_rot = self.rotations.jacobian_translation(
                                                      point=rot_vec,
                                                      left_or_right='left')
            jacobian_trans = self.rotations.matrix_from_rotation_vector(
                    rot_vec)

            jacobian[:, :3, :3] = jacobian_rot
            jacobian[:, 3:, 3:] = jacobian_trans

        else:
            jacobian_rot = self.rotations.jacobian_translation(
                                                      point=rot_vec,
                                                      left_or_right='right')
            jacobian[:, :3, :3] = jacobian_rot
            jacobian[:, 3:, :3] = - so_group.skew_matrix_from_vector(rot_vec)
            jacobian[:, 3:, 3:] = np.eye(3)

        assert jacobian.ndim == 3
        return jacobian

    def group_exp_from_identity(self,
                                tangent_vec):
        """
        Compute the group exponential of vector tangent_vector,
        at point base_point.

        :param tangent_vector: tangent vector of SE(3) at base_point.
        :param base_point: 6d vector element of SE(3).
        :returns group_exp_transfo: 6d vector element of SE(3).
        """
        if tangent_vec.ndim == 1:
            tangent_vec = np.expand_dims(tangent_vec, axis=0)
        assert tangent_vec.ndim == 2
        rot_vec = tangent_vec[:, 0:3]
        rot_vec = self.rotations.regularize(rot_vec)
        translation = tangent_vec[:, 3:6]
        angle = np.linalg.norm(rot_vec)

        if np.isclose(angle, np.pi):
            rot_vec = self.rotations.regularize(rot_vec)

        group_exp_transfo = np.zeros_like(tangent_vec)
        group_exp_transfo[:, 0:3] = rot_vec

        skew_mat = so_group.skew_matrix_from_vector(rot_vec)

        if angle == 0:
            coef_1 = 0
            coef_2 = 0
        elif np.isclose(angle, 0):
            coef_1 = 1. / 2. - angle ** 2 / 24. + angle ** 4 / 720.
            coef_2 = 1. / 6 - angle ** 2 / 120. + angle ** 4 / 5040.

        else:
            coef_1 = (1. - np.cos(angle)) / angle ** 2
            coef_2 = (angle - np.sin(angle)) / angle ** 3

        sq_skew_mat = np.matmul(skew_mat, skew_mat)
        term_1 = coef_1 * np.dot(translation,
                                 np.transpose(skew_mat, axes=(0, 2, 1)))
        term_2 = coef_2 * np.dot(translation,
                                 np.transpose(sq_skew_mat, axes=(0, 2, 1)))

        group_exp_transfo[:, 3:6] = translation + term_1 + term_2


        group_exp_transfo = self.regularize(group_exp_transfo)
        return group_exp_transfo

    def group_log_from_identity(self,
                                point):
        """
        Compute the group logarithm of point point,
        from the identity.
        """
        ##Vectorized
        assert self.belongs(point)
        point = self.regularize(point)

        rot_vec = point[:, 0:3]
        angle = np.linalg.norm(rot_vec)
        translation = point[:, 3:6]

        group_log = np.zeros_like(point)
        group_log[:, 0:3] = rot_vec
        skew_rot_vec = so_group.skew_matrix_from_vector(rot_vec)
        sq_skew_rot_vec = np.matmul(skew_rot_vec, skew_rot_vec)

        if angle == 0:
            coef_1 = 0
            coef_2 = 0

        elif np.isclose(angle, 0):
            coef_1 = - 0.5
            coef_2 = 0.5 - angle ** 2 / 90

        elif np.isclose(angle, np.pi):
            delta_angle = angle - np.pi
            coef_1 = - 0.5
            psi = 0.5 * angle * (- delta_angle / 2. - delta_angle ** 3 / 24.)
            coef_2 = (1 - psi) / (angle ** 2)

        else:
            coef_1 = - 0.5
            psi = 0.5 * angle * np.sin(angle) / (1 - np.cos(angle))
            coef_2 = (1 - psi) / (angle ** 2)

        term_1 = coef_1 * np.dot(translation,
                                 np.transpose(skew_rot_vec, axes=(0, 2, 1)))
        term_2 = coef_2 * np.dot(translation,
                                 np.transpose(sq_skew_rot_vec, axes=(0, 2, 1)))

        group_log[:, 3:6] = translation + term_1 + term_2

        assert group_log.ndim == 2

        return group_log

    def random_uniform(self):
        """
        Generate an 6d vector element of SE(3) uniformly,
        by generating separately a rotation vector uniformly
        on the hypercube of sides [-1, 1] in the tangent space,
        and a translation in the hypercube of side [-1, 1] in
        the euclidean space.
        """
        random_rot_vec = self.rotations.random_uniform()
        random_translation = self.translations.random_uniform()

        random_transfo = np.concatenate([random_rot_vec, random_translation],
                                        axis=1)
        random_transfo = self.regularize(random_transfo)
        return random_transfo

    def exponential_matrix(self, rot_vec):
        """
        Compute the exponential of the rotation matrix
        represented by rot_vec.

        :param rot_vec: 3D rotation vector
        :returns exponential_mat: 3x3 matrix
        """

        rot_vec = self.rotations.regularize(rot_vec)

        angle = np.linalg.norm(rot_vec)
        skew_rot_vec = so_group.skew_matrix_from_vector(rot_vec)

        if angle == 0:
            coef_1 = 0
            coef_2 = 0

        elif np.isclose(angle, 0):
            coef_1 = 1. / 2. - angle ** 2 / 24.
            coef_2 = 1. / 6. - angle ** 3 / 120.

        else:
            coef_1 = angle ** (-2) * (1. - np.cos(angle))
            coef_2 = angle ** (-2) * (1. - np.sin(angle) / angle)

        identity = np.repeat(np.identity(3), repeats=rot_vec.shape[0])
        exponential_mat = (identity
                           + coef_1 * skew_rot_vec
                           + coef_2 * np.matmul(skew_rot_vec, skew_rot_vec))

        return exponential_mat

    def group_exponential_barycenter(self, points, weights=None):
        """
        Compute the group exponential barycenter.

        :param points: SE3 data points, Nx6 array
        :param weights: data point weights, Nx1 array
        """

        n_points = points.shape[0]
        assert n_points > 0

        if weights is None:
            weights = np.ones((n_points, 1))

        n_weights = weights.shape[0]
        assert n_points == n_weights

        dim_rotations = self.rotations.dimension
        dim = self.dimension

        rotation_vectors = points[:, :dim_rotations]
        translations = points[:, dim_rotations:dim]
        assert rotation_vectors.shape == (n_points, dim_rotations)
        assert translations.shape == (n_points, self.n)

        mean_rotation = self.rotations.group_exponential_barycenter(
                                                points=rotation_vectors,
                                                weights=weights)
        mean_rotation_mat = self.rotations.matrix_from_rotation_vector(
                    mean_rotation)

        matrix = np.zeros([1, self.n, self.n])
        translation_aux = np.zeros([1, self.n])

        inv_rot_mats = self.rotations.matrix_from_rotation_vector(
                -rotation_vectors)
        # TODO(nina): this is the same mat multiplied several times
        matrix_aux = np.matmul(mean_rotation_mat, inv_rot_mats)
        assert matrix_aux.shape == (n_points, dim_rotations, dim_rotations)

        vec_aux = self.rotations.rotation_vector_from_matrix(matrix_aux)
        matrix_aux = self.exponential_matrix(vec_aux)
        matrix_aux = np.linalg.inv(matrix_aux)

        matrix += weights * matrix_aux

        translation_aux += weights * np.dot(translations,
                                            np.transpose(np.matmul(
                                                matrix_aux,
                                                inv_rot_mats), axes=(0, 2, 1)))

        mean_translation = np.dot(translation_aux,
                                  np.transpose(np.linalg.inv(matrix),
                                               axes=(0, 2, 1)))

        exp_bar = np.zeros([1, dim])
        exp_bar[1, :dim_rotations] = mean_rotation
        exp_bar[1, dim_rotations:dim] = mean_translation

        return exp_bar
