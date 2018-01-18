"""Computations on the Lie group of 3D rotations."""

import numpy as np

import geomstats.LieGroup as LieGroup
ALGEBRA_CANONICAL_INNER_PRODUCT = np.eye(3)


def closest_rotation_matrix(mat):
    """
    Compute the closest - in terms of
    the Frobenius norm - rotation matrix
    of a given matrix M.
    This avoids computational errors.

    :param mat: 3x3 matrix
    :returns rot_mat: 3x3 rotation matrix.
    """
    assert mat.shape == (3, 3)

    mat_unitary_u, mat_diag_s, mat_unitary_v = np.linalg.svd(mat)
    rot_mat = np.dot(mat_unitary_u, mat_unitary_v)
    mat_diag_s = np.eye(3) * mat_diag_s

    if np.linalg.det(rot_mat) < 0:
        mat_diag_s[0, 0] = 1
        mat_diag_s[1, 1] = 1
        mat_diag_s[2, 2] = -1
        rot_mat = np.dot(np.dot(mat_unitary_u, mat_diag_s),
                         mat_unitary_v)

    return rot_mat


def skew_matrix_from_vector(vec):
    """
    Compute the skew-symmetric matrix,
    known as the cross-product of a vector,
    associated to the vector vec.

    :param vec: 3d vector
    :returns skew_vec: 3x3 skew-symmetric matrix
    """
    assert len(vec) == 3

    skew_vec = np.array([[0, -vec[2], vec[1]],
                         [vec[2], 0, -vec[0]],
                         [-vec[1], vec[0], 0]])
    return skew_vec


class SpecialOrthogonalGroup(LieGroup):

    def __init__(self, dimension):
        if dimension is not 3:
            raise NotImplementedError('Only SO(3) is implemented.')
        self.identity = np.array([0., 0., 0.])

    def regularize(self, rot_vec):
        """
        Regularize the norm of the rotation vector,
        to be between 0 and pi, following the axis-angle
        representation's convention.

        If the angle angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        :param rot_vec: 3d vector
        :returns regularized_rot_vec: 3d vector with: 0 < norm < pi
        """
        assert len(rot_vec) == 3

        rot_vec = np.array(rot_vec)
        angle = np.linalg.norm(rot_vec)

        regularized_rot_vec = rot_vec
        if angle != 0:
            k = np.floor(angle / (2 * np.pi) + .5)
            regularized_rot_vec = (1. - 2. * np.pi * k / angle) * rot_vec

        return regularized_rot_vec

    def rotation_vector_from_rotation_matrix(self, rot_mat, epsilon=1e-5):
        """
        Convert rotation matrix to rotation vector
        (axis-angle representation).

        :param rot_mat: 3x3 rotation matrix
        :returns rot_vec: 3d rotation vector
        """
        assert rot_mat.shape == (3, 3)

        # -- Get angle, angle of the rotation
        # Note: trace(rotation_matrix) = 2cos(angle) + 1
        cos_angle = np.clip((np.trace(rot_mat) - 1) * 0.5, -1, 1)
        angle = np.arccos(cos_angle)

        # -- Edge case: angle close to 0
        if angle < epsilon:
            coef = 0.5 * (1 + (angle ** 2) / 6)
            # Note: Taylor expansion of sin(angle) around angle = 0
            skew_rot_vec = coef * (rot_mat - rot_mat.transpose())
            rot_vec = np.array([skew_rot_vec[2][1],
                                skew_rot_vec[0][2],
                                skew_rot_vec[1][0]])
        # -- Edge case: angle close to pi
        elif abs(angle - np.pi) < epsilon:
            rot_vec = np.empty(3)
            for i in range(0, 3):
                sq_element_i = 1 + (rot_mat[i][i] - 1) / (1 - cos_angle)
                sq_element_i = np.clip(sq_element_i, 0, 1)
                rot_vec[i] = np.sqrt(sq_element_i)

            rot_vec = rot_vec * angle / np.linalg.norm(rot_vec)
            if rot_mat[0][1] + rot_mat[1][0] < 0:
                rot_vec[1] = -rot_vec[1]
            if rot_mat[0][2] + rot_mat[2][0] < 0:
                rot_vec[2] = -rot_vec[2]
            sinr = np.zeros((3,))
            sinr[0] = rot_mat[2][1] - rot_mat[1][2]
            sinr[1] = rot_mat[0][2] - rot_mat[2][0]
            sinr[2] = rot_mat[1][0] - rot_mat[0][1]

            k = 0
            if abs(sinr[1]) > abs(sinr[k]):
                k = 1
            if abs(sinr[2]) > abs(sinr[k]):
                k = 2
            if sinr[k] * rot_vec[k] < 0:
                rot_vec = -rot_vec
        # -- Regular case (see wikipedia wrt math computations)
        else:
            coef = .5 * angle / np.sin(angle)
            skew_rot_vec = coef * (rot_mat - rot_mat.transpose())
            rot_vec = np.array([skew_rot_vec[2][1],
                                skew_rot_vec[0][2],
                                skew_rot_vec[1][0]])
        return regularize(rot_vec)

    def rotation_matrix_from_rotation_vector(self, rot_vec, epsilon=1e-5):
        """
        Convert rotation vector to rotation matrix.

        :param rot_vec: 3d rotation vector
        :returns rot_mat: 3x3 rotation matrix

        """
        assert len(rot_vec) == 3
        rot_vec = regularize(rot_vec)

        angle = np.linalg.norm(rot_vec)
        skew_rot_vec = skew_matrix_from_vector(rot_vec)

        if angle < epsilon:
            sin_angle = 1 - (angle ** 2) / 6
            cos_angle = 1 / 2 - angle ** 2
        else:
            sin_angle = np.sin(angle) / angle
            cos_angle = (1 - np.cos(angle)) / (angle ** 2)

        rot_mat = (np.identity(3) + sin_angle * skew_rot_vec
                   + cos_angle * np.dot(skew_rot_vec, skew_rot_vec))
        return rot_mat


    def compose(self, rot_vec_1, rot_vec_2):
        """
        Compose 2 rotation vectors according to the matrix product
        on the corresponding matrices.
        """
        rot_vec_1 = regularize(rot_vec_1)
        rot_vec_2 = regularize(rot_vec_2)

        rot_mat_1 = rotation_matrix_from_rotation_vector(rot_vec_1)
        rot_mat_2 = rotation_matrix_from_rotation_vector(rot_vec_2)

        rot_mat_prod = np.matmul(rot_mat_1, rot_mat_2)
        rot_vec_prod = rotation_vector_from_rotation_matrix(rot_mat_prod)

        return rot_vec_prod


    def jacobian_translation(self, rot_vec,
                             left_or_right='left', epsilon=1e-5):
        """
        Compute the jacobian matrix of the differential
        of the left translation by the rotation r.

        :param rot_vec: 3D rotation vector
        :returns jacobian: 3x3 matrix
        """
        assert len(rot_vec) == 3
        assert left_or_right in ('left', 'right')
        rot_vec = regularize(rot_vec)

        angle = np.linalg.norm(rot_vec)
        if angle < epsilon:
            coef_1 = 1 - angle ** 2 / 12
            coef_2 = 1 / 12 + angle ** 2 / 720
        elif abs(angle - np.pi) < epsilon:
            coef_1 = angle * (np.pi - angle) / 4
            coef_2 = (1 - coef_1) / angle ** 2
        else:
            coef_1 = (angle / 2) / np.tan(angle / 2)
            coef_2 = (1 - coef_1) / angle ** 2

        if left_or_right == 'left':
            jacobian = (coef_1 * np.identity(3)
                        + coef_2 * np.outer(rot_vec, rot_vec)
                        + skew_matrix_from_vector(rot_vec) / 2)

        else:
            jacobian = (coef_1 * np.identity(3)
                        + coef_2 * np.outer(rot_vec, rot_vec)
                        - skew_matrix_from_vector(rot_vec) / 2)

        return jacobian


    def riemannian_log(rot_vec,
                       inner_product=ALGEBRA_CANONICAL_INNER_PRODUCT,
                       left_or_right='left',
                       ref_point=GROUP_IDENTITY):
        """
        Compute the Riemannian logarithm at point ref_point,
        of point rot_vec wrt the metric obtained by left/right
        translation of the inner product inner_product at
        the Lie algebra.

        This gives a tangent vector at point ref_point.

        :param rot_vec: 3D rotation vector
        :param ref_point: 3D rotation vector
        :returns rot_vec_log: 3D rotation vector, tangent vector at ref_point
        """
        assert len(rot_vec) == 3 & len(ref_point) == 3
        assert left_or_right in ('left', 'right')
        rot_vec = regularize(rot_vec)
        ref_point = regularize(ref_point)

        if left_or_right == 'left':
            rot_mat_ref_inv = rotation_matrix_from_rotation_vector(-ref_point)
            rot_mat = rotation_matrix_from_rotation_vector(rot_vec)
            rot_mat_prod = np.matmul(rot_mat_ref_inv, rot_mat)
            rot_vec_translated = rotation_vector_from_rotation_matrix(rot_mat_prod)

            jacobian = jacobian_translation(ref_point, left_or_right='left')
            rot_vec_log = np.dot(jacobian, rot_vec_translated)

        else:
            raise NotImplementedError()

        return rot_vec_log
