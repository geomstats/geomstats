"""Computations on the Lie group of 3D rotations."""

import numpy as np

from geomstats.lie_group import LieGroup

EPSILON = 1e-5


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
    :return skew_mat: 3x3 skew-symmetric matrix
    """
    assert len(vec) == 3

    skew_mat = np.array([[0, -vec[2], vec[1]],
                         [vec[2], 0, -vec[0]],
                         [-vec[1], vec[0], 0]])
    return skew_mat


def vector_from_skew_matrix(skew_mat):
    """
    Compute the skew-symmetric matrix,
    known as the cross-product of a vector,
    associated to the vector vec.

    :param skew_mat: 3x3 skew-symmetric matrix
    :return vec: 3d vector
    """
    assert skew_mat.shape == (3, 3)

    vec = np.array([skew_mat[2][1],
                    skew_mat[0][2],
                    skew_mat[1][0]])
    return vec


class SpecialOrthogonalGroup(LieGroup):

    def __init__(self, n):
        assert n > 1

        if n is not 3:
            raise NotImplementedError('Only SO(3) is implemented.')

        self.n = n
        self.dimension = int((n * (n - 1)) / 2)
        super(SpecialOrthogonalGroup, self).__init__(
                          dimension=self.dimension,
                          identity=np.zeros(self.dimension))
        self.bi_invariant_metric = self.left_canonical_metric

    def belongs(self, rot_vec):
        """
        Check that a vector belongs to the
        special orthogonal group.
        """
        return len(rot_vec) == self.dimension

    def regularize(self, rot_vec):
        """
        Regularize the norm of the rotation vector,
        to be between 0 and pi, following the axis-angle
        representation's convention.

        If the angle angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        :param rot_vec: 3d vector
        :returns self.regularized_rot_vec: 3d vector with: 0 < norm < pi
        """
        assert self.belongs(rot_vec)

        rot_vec = np.array(rot_vec)
        angle = np.linalg.norm(rot_vec)

        # self.regularized_rot_vec = rot_vec
        # if angle != 0:
        #     k = np.floor(angle / (2 * np.pi) + .5)
        #     self.regularized_rot_vec = (1. - 2. * np.pi * k / angle) * rot_vec
        regularized_rot_vec = rot_vec
        if angle != 0:
            k = np.floor(angle / (np.pi))
            sign = 1 if k % 2 == 0 else -1
            regularized_rot_vec = sign * (1. - np.pi * k / angle) * rot_vec

        return regularized_rot_vec

    def rotation_vector_from_matrix(self, rot_mat, epsilon=EPSILON):
        """
        Convert rotation matrix to rotation vector
        (axis-angle representation).

        :param rot_mat: 3x3 rotation matrix
        :returns rot_vec: 3d rotation vector
        """
        assert rot_mat.shape == (self.n, self.n)

        cos_angle = .5 * (np.trace(rot_mat) - 1)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        # -- Edge case: angle close to 0
        if angle < epsilon:
            # Taylor expansion of 0.5 * angle / sin(angle) around 0:
            coef = 0.5 * (1 + (angle ** 2) / 6)
            skew_rot_vec = coef * (rot_mat - rot_mat.transpose())
            rot_vec = vector_from_skew_matrix(skew_rot_vec)

        # -- Edge case: angle close to pi
        elif abs(angle - np.pi) < epsilon:
            rot_vec = np.empty(self.dimension)
            diag_rot_mat = np.diag(rot_mat)
            sq_element = 1 + (diag_rot_mat - np.ones(self.n)) / (1 - cos_angle)
            sq_element = np.clip(sq_element, 0, 1)
            rot_vec = np.sqrt(sq_element)

            rot_vec = rot_vec * angle / np.linalg.norm(rot_vec)
            if rot_mat[0][1] + rot_mat[1][0] < 0:
                rot_vec[1] = -rot_vec[1]
            if rot_mat[0][2] + rot_mat[2][0] < 0:
                rot_vec[2] = -rot_vec[2]

            sinr = np.zeros(self.dimension)
            aux_mat = rot_mat - rot_mat.transpose()
            sinr = vector_from_skew_matrix(aux_mat)

            k = 0
            if abs(sinr[1]) > abs(sinr[k]):
                k = 1
            if abs(sinr[2]) > abs(sinr[k]):
                k = 2
            if sinr[k] * rot_vec[k] < 0:
                rot_vec = -rot_vec

        else:
            coef = .5 * angle / np.sin(angle)
            skew_rot_vec = coef * (rot_mat - rot_mat.transpose())
            rot_vec = vector_from_skew_matrix(skew_rot_vec)
        return self.regularize(rot_vec)

    def matrix_from_rotation_vector(self, rot_vec, epsilon=EPSILON):
        """
        Convert rotation vector to rotation matrix.

        :param rot_vec: 3d rotation vector
        :returns rot_mat: 3x3 rotation matrix

        """
        assert self.belongs(rot_vec)
        rot_vec = self.regularize(rot_vec)

        angle = np.linalg.norm(rot_vec)
        skew_rot_vec = skew_matrix_from_vector(rot_vec)

        if angle < epsilon:
            coef_1 = 1 - (angle ** 2) / 6
            coef_2 = 1 / 2 - angle ** 2
        else:
            coef_1 = np.sin(angle) / angle
            coef_2 = (1 - np.cos(angle)) / (angle ** 2)

        rot_mat = (np.identity(self.dimension)
                   + coef_1 * skew_rot_vec
                   + coef_2 * np.dot(skew_rot_vec, skew_rot_vec))
        return rot_mat

    def compose(self, rot_vec_1, rot_vec_2):
        """
        Compose 2 rotation vectors according to the matrix product
        on the corresponding matrices.
        """
        rot_vec_1 = self.regularize(rot_vec_1)
        rot_vec_2 = self.regularize(rot_vec_2)

        rot_mat_1 = self.matrix_from_rotation_vector(rot_vec_1)
        rot_mat_2 = self.matrix_from_rotation_vector(rot_vec_2)

        rot_mat_prod = np.matmul(rot_mat_1, rot_mat_2)
        rot_vec_prod = self.rotation_vector_from_matrix(rot_mat_prod)

        return rot_vec_prod

    def inverse(self, rot_vec):
        """
        Inverse of a rotation.
        """
        return -rot_vec

    def jacobian_translation(self, point,
                             left_or_right='left', epsilon=EPSILON):
        """
        Compute the jacobian matrix of the differential
        of the left translation by the rotation r.

        :param rot_vec: 3D rotation vector
        :returns jacobian: 3x3 matrix
        """
        assert self.belongs(point)
        assert left_or_right in ('left', 'right')
        point = self.regularize(point)

        angle = np.linalg.norm(point)
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
            jacobian = (coef_1 * np.identity(self.dimension)
                        + coef_2 * np.outer(point, point)
                        + skew_matrix_from_vector(point) / 2)

        else:
            jacobian = (coef_1 * np.identity(self.dimension)
                        + coef_2 * np.outer(point, point)
                        - skew_matrix_from_vector(point) / 2)

        return jacobian

    def random_uniform(self):
        """
        Sample a 3d rotation vector uniformly, w.r.t.
        the bi-invariant metric, by sampling in the
        hypercube of side [-1, 1] on the tangent space.
        """
        random_rot_vec = np.random.rand(self.dimension) * 2 - 1
        random_rot_vec = self.regularize(random_rot_vec)
        return random_rot_vec

    def group_exp_from_identity(self, tangent_vec):
        """
        Compute the group exponential of vector tangent_vector.
        """
        tangent_vec = self.regularize(tangent_vec)
        return tangent_vec

    def group_log_from_identity(self, point):
        """
        Compute the group logarithm of point point.
        """
        point = self.regularize(point)
        return point

    def group_exponential_barycenter(self, points, weights=None):
        """
        Group exponential barycenter is the Frechet mean
        of the bi-invariant metric.
        """
        n_points = len(points)
        assert n_points > 0

        if weights is None:
            weights = np.ones(n_points)

        n_weights = len(weights)
        assert n_points == n_weights

        barycenter = self.bi_invariant_metric.mean(points, weights)

        return barycenter
