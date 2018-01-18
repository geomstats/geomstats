"""Computations on the Lie group of rigid transformations."""

import numpy as np

import geomstats.special_orthogonal_group as so_group

from geomstats.euclidean_space import EuclideanSpace
from geomstats.lie_groups import LieGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

EPSILON = 1e-5
ALGEBRA_CANONICAL_INNER_PRODUCT = np.eye(6)


class SpecialEuclideanGroup(LieGroup):

    def __init__(self, dimension):
        if dimension is not 3:
            raise NotImplementedError('Only SE(3) is implemented.')
        self.rotations = SpecialOrthogonalGroup.__init__(dimension)
        self.translations = EuclideanSpace.__init__(dimension)

        self.identity = np.concatenate([self.rotations.identity,
                                        np.array([0., 0., 0.])])

    def inner_product(self, coef_rotations, coef_translations):
        """
        Compute a 6x6 diagonal matrix, where the diagonal is formed by:
        coef_rotations * [1, 1, 1] and coef_translations * [1, 1, 1].

        :param coef_rotations: scalar
        :param coef_translations: scalar
        :returns inner_product_mat: 6x6 matrix
        """
        inner_product_mat = np.zeros([6, 6])
        inner_product_mat[0:3, 0:3] = coef_rotations * np.eye(3)
        inner_product_mat[3:6, 3:6] = coef_translations * np.eye(3)
        return inner_product_mat

    def regularize(self, transfo):
        """
        Regularize an element of the group SE(3),
        by extracting the rotation vector r from the input [r t]
        and using self.rotations.regularize.

        :param transfo: 6d vector, element in SE(3) represented as [r t].
        :returns self.regularized_transfo: 6d vector, element in SE(3)
        with self.regularized rotation.
        """

        regularized_transfo = transfo
        rot_vec = transfo[0:3]
        regularized_transfo[0:3] = self.rotations.regularize(rot_vec)

        return regularized_transfo

    def inverse(self, transfo):
        """
        Compute the group inverse in SE(3).

        Formula:
        (R, t)^{-1} = (R^{-1}, R^{-1}.(-t))

        :param transfo: 6d vector element in SE(3)
        :returns inverse_transfo: 6d vector inverse of transfo
        """

        transfo = self.regularize(transfo)

        rot_vec = transfo[0:3]
        translation = transfo[3:6]

        inverse_transfo = np.zeros(6)
        inverse_transfo[0:3] = -rot_vec
        rot_mat = self.rotations.matrix_from_rotation_vector(-rot_vec)
        inverse_transfo[3:6] = np.dot(rot_mat, -translation)

        inverse_transfo = self.regularize(inverse_transfo)
        return inverse_transfo

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
        rot_mat_1 = self.rotations.matrix_from_rotation_vector(transfo_1[0:3])
        rot_mat_1 = self.rotations.closest_rotation_matrix(rot_mat_1)

        rot_mat_2 = self.rotations.matrix_from_rotation_vector(transfo_2[0:3])
        rot_mat_2 = self.rotations.closest_rotation_matrix(rot_mat_2)

        translation_1 = transfo_1[3:6]
        translation_2 = transfo_2[3:6]

        prod_transfo = np.zeros(6)
        prod_rot_mat = np.dot(rot_mat_1, rot_mat_2)

        prod_transfo[:3] = self.rotations.rotation_vector_from_matrix(
                                                                  prod_rot_mat)
        prod_transfo[3:6] = np.dot(rot_mat_1, translation_2) + translation_1

        prod_transfo = self.regularize(prod_transfo)
        return prod_transfo

    def jacobian_translation(self, transfo, left_or_right='left'):
        """
        Compute the jacobian matrix of the differential
        of the left/right translations
        from the identity to transfo in the Lie group SE(3).

        :param transfo: 6D vector element of SE(3)
        :returns jacobian: 6x6 matrix
        """
        assert len(transfo) == 6
        assert left_or_right in ('left', 'right')

        transfo = self.regularize(transfo)

        rot_vec = transfo[0:3]
        translation = transfo[3:6]

        jacobian = np.zeros((6, 6))

        if left_or_right == 'left':
            jacobian_rot = self.rotations.jacobian_translation(
                                                      rot_vec,
                                                      left_or_right='left')
            jacobian_trans = self.rotations.matrix_from_rotation_vector(
                    rot_vec)

            jacobian[:3, :3] = jacobian_rot
            jacobian[3:, 3:] = jacobian_trans

        else:
            jacobian_rot = self.rotations.jacobian_translation(
                                                      rot_vec,
                                                      left_or_right='right')
            jacobian[:3, :3] = jacobian_rot
            jacobian[3:, :3] = - so_group.skew_matrix_from_vector(translation)
            jacobian[3:, 3:] = np.eye(3)

        return jacobian

    def group_exp(self,
                  tangent_vector,
                  ref_point,
                  epsilon=EPSILON):
        """
        Compute the group exponential of vector tangent_vector,
        at point ref_point.

        :param tangent_vector: tangent vector of SE(3) at ref_point.
        :param ref_point: 6d vector element of SE(3).
        :returns group_exp_transfo: 6d vector element of SE(3).
        """
        tangent_vector = self.regularize(tangent_vector)

        if ref_point is self.identity:
            rot_vec = tangent_vector[0:3]
            translation = tangent_vector[3:6]  # this is dt
            angle = np.linalg.norm(rot_vec)

            group_exp_transfo = np.zeros(6)
            group_exp_transfo[0:3] = rot_vec

            skew_mat = so_group.skew_matrix_from_vector(rot_vec)

            if angle == 0:
                coef_1 = 0
                coef_2 = 0
            elif angle < epsilon:
                coef_1 = 1. / 2. - angle ** 2 / 12.
                coef_2 = 1. - angle ** 3 / 3.
            else:
                coef_1 = (1. - np.cos(angle)) / angle ** 2
                coef_2 = (angle - np.sin(angle)) / angle ** 3

            sq_skew_mat = np.dot(skew_mat, skew_mat)
            group_exp_transfo[3:6] = (translation
                                      + coef_1 * np.dot(skew_mat,
                                                        translation)
                                      + coef_2 * np.dot(sq_skew_mat,
                                                        translation))

        else:
            ref_point = self.regularize(ref_point)

            jacobian = self.jacobian_translation(ref_point,
                                                 left_or_right='left')
            inv_jacobian = np.linalg.inv(jacobian)

            tangent_vector_at_identity = np.dot(inv_jacobian, tangent_vector)
            group_exp_from_identity = self.group_exp(
                                                  tangent_vector_at_identity)

            group_exp_transfo = self.compose(ref_point,
                                             group_exp_from_identity)

        return self.regularize(group_exp_transfo)

    def group_log(self, transfo,
                  ref_point,
                  epsilon=EPSILON):
        """
        Compute the group logarithm of point transfo,
        from point ref_point.

        :param transfo: 6d tangent_vector element of SE(3)
        :param ref_point: 6d tangent_vector element of SE(3)

        :returns tangent vector: 6d tangent vector at ref_point.
        """

        transfo = self.regularize(transfo)
        if ref_point is self.identity:
            rot_vec = transfo[0:3]
            angle = np.linalg.norm(rot_vec)
            translation = transfo[3:6]

            tangent_vector = np.zeros(6)
            tangent_vector[0:3] = rot_vec
            skew_rot_vec = so_group.skew_matrix_from_vector(rot_vec)
            sq_skew_rot_vec = np.dot(skew_rot_vec, skew_rot_vec)

            if angle == 0:
                coef_1 = 0
                coef_2 = 0
                tangent_vector[3:6] = translation

            elif angle < epsilon:
                coef_1 = - 0.5
                coef_2 = 0.5 - angle ** 2 / 90

            else:
                coef_1 = - 0.5
                psi = 0.5 * angle * np.sin(angle) / (1 - np.cos(angle))
                coef_2 = (1 - psi) / (angle ** 2)

            tangent_vector[3:6] = (translation
                                   + coef_1 * np.dot(skew_rot_vec, translation)
                                   + coef_2 * np.dot(sq_skew_rot_vec,
                                                     translation))
        else:
            ref_point = self.regularize(ref_point)
            jacobian = self.jacobiance_translation(ref_point,
                                                   left_or_right='left')
            transfo_near_id = self.compose(self.inverse(ref_point), transfo)
            tangent_vector_from_id = self.group_log(transfo_near_id)
            tangent_vector = np.dot(jacobian, tangent_vector_from_id)

        return tangent_vector

    def random_uniform(self):
        """
        Generate an 6d vector element of SE(3) uniformly

        :returns random transfo: 6d vector element of SE(3)
        """
        random_rot_vec = np.random.rand(3) * 2 - 1
        random_rot_vec = self.rotations.regularize(random_rot_vec)
        random_translation = np.random.rand(3) * 2 - 1

        random_transfo = np.concatenate([random_rot_vec, random_translation])
        return random_transfo

    def exponential_matrix(self, rot_vec, epsilon=1e-5):
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

        elif angle < epsilon:
            coef_1 = 1. / 2. - angle ** 2 / 24.
            coef_2 = 1. / 6. - angle ** 3 / 120.

        else:
            coef_1 = angle ** (-2) * (1. - np.cos(angle))
            coef_2 = angle ** (-2) * (1. - np.sin(angle) / angle)

        exponential_mat = (np.eye(3)
                           + coef_1 * skew_rot_vec
                           + coef_2 * np.dot(skew_rot_vec, skew_rot_vec))

        return exponential_mat

    def exponential_barycenter(self, transfo_vectors, weights):
        """

        :param transfo_vectors: SE3 data points, Nx6 array
        :param weights: data point weights, Nx1 array
        """

        n_transformations, _ = transfo_vectors.shape

        if n_transformations < 2:
            raise ValueError('Requires # of transformations >=2.')

        rotation_vectors = transfo_vectors[:, 0:3]
        translations = transfo_vectors[:, 3:6]

        biinvariant_metric = so_group.InvariantMetric(self.rotations,
                                                      np.eye(3))
        mean_rotation = biinvariant_metric.riemannian_mean(rotation_vectors,
                                                           weights)

        # Partie translation, p34 de expbar
        matrix = np.zeros([3, 3])
        translation = np.zeros(3)

        for i in range(0, n_transformations):
            rot_vec_i = rotation_vectors[i, :]
            translation_i = translations[i, :]
            weight_i = weights[i]

            inv_rot_mat_i = so_group.matrix_from_rotation_vector(
                    -rot_vec_i)
            mean_rotation_mat = so_group.matrix_from_rotation_vector(
                    mean_rotation)

            matrix_aux = np.dot(mean_rotation_mat, inv_rot_mat_i)
            vec_aux = so_group.rotation_vector_from_matrix(matrix_aux)
            matrix_aux = self.exponential_matrix(vec_aux)
            matrix_aux = np.linalg.inv(matrix_aux)

            matrix = matrix + weight_i * matrix_aux
            translation = (translation
                           + weight_i * np.dot(np.dot(matrix_aux,
                                                      inv_rot_mat_i),
                                               translation_i))

        mean_transformation = np.zeros(6)
        mean_transformation[0:3] = mean_rotation
        mean_transformation[3:6] = np.dot(np.linalg.inv(matrix), translation)
        return mean_transformation
