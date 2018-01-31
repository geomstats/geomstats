"""Computations on the Lie group of 3D rotations."""

import numpy as np

from geomstats.lie_group import LieGroup


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
    vec = skew_mat[(2, 0, 1), (1, 2, 0)]
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
        angle = np.linalg.norm(rot_vec)
        regularized_rot_vec = rot_vec

        if angle != 0:
            k = np.floor(angle / (2 * np.pi) + .5)
            regularized_rot_vec = (1. - 2. * np.pi * k / angle) * rot_vec

        return regularized_rot_vec

    def rotation_vector_from_matrix(self, rot_mat):
        """
        Convert rotation matrix to rotation vector
        (axis-angle representation).

        Get the angle through the trace of the rotation matrix:
        The eigenvalues are:
        1, cos(angle) + i sin(angle), cos(angle) - i sin(angle)
        so that: trace = 1 + 2 cos(angle), -1 <= trace <= 3

        Get the rotation vector through the formula:
        S_r = angle / ( 2 * sin(angle) ) (R - R^T)

        For the edge case where the angle is close to pi,
        the formulation is derived by going from rotation matrix to unit
        quaternion to axis-angle:
         r = angle * v / |v|, where (w, v) is a unit quaternion.

        :param rot_mat: 3x3 rotation matrix
        :return rot_vec: 3d rotation vector
        """
        assert rot_mat.shape == (3, 3)

        rot_mat = closest_rotation_matrix(rot_mat)

        trace = np.trace(rot_mat)
        cos_angle = .5 * (trace - 1)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        rot_vec = vector_from_skew_matrix(rot_mat - rot_mat.transpose())

        if np.isclose(angle, 0):
            rot_vec = (.5 - (trace - 3.) / 12.) * rot_vec
        elif np.isclose(angle, np.pi):
            # choose the largest diagonal element
            # to avoid a square root of a negative number
            a = np.argmax(np.diag(rot_mat))
            b = np.mod(a + 1, 3)
            c = np.mod(a + 2, 3)

            # compute the axis vector
            sq_root = np.sqrt((rot_mat[a, a]
                              - rot_mat[b, b] - rot_mat[c, c] + 1.))

            rot_vec = np.zeros(3)
            rot_vec[a] = sq_root / 2.
            rot_vec[b] = (rot_mat[b, a] + rot_mat[a, b]) / (2. * sq_root)
            rot_vec[c] = (rot_mat[c, a] + rot_mat[a, c]) / (2. * sq_root)

            rot_vec = angle * rot_vec / np.linalg.norm(rot_vec)

        else:
            rot_vec = angle / (2. * np.sin(angle)) * rot_vec

        return self.regularize(rot_vec)

    def matrix_from_rotation_vector(self, rot_vec):
        """
        Convert rotation vector to rotation matrix.

        :param rot_vec: 3d rotation vector
        :returns rot_mat: 3x3 rotation matrix

        """
        assert self.belongs(rot_vec)
        rot_vec = self.regularize(rot_vec)

        angle = np.linalg.norm(rot_vec)
        skew_rot_vec = skew_matrix_from_vector(rot_vec)

        if np.isclose(angle, 0):
            coef_1 = 1 - (angle ** 2) / 6
            coef_2 = 1 / 2 - angle ** 2
        else:
            coef_1 = np.sin(angle) / angle
            coef_2 = (1 - np.cos(angle)) / (angle ** 2)

        rot_mat = (np.identity(self.dimension)
                   + coef_1 * skew_rot_vec
                   + coef_2 * np.dot(skew_rot_vec, skew_rot_vec))
        rot_mat = closest_rotation_matrix(rot_mat)
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
                             left_or_right='left'):
        """
        Compute the jacobian matrix of the differential
        of the left translation by the rotation r.

        Formula:
        https://hal.inria.fr/inria-00073871

        :param rot_vec: 3D rotation vector
        :returns jacobian: 3x3 matrix
        """
        assert self.belongs(point)
        assert left_or_right in ('left', 'right')
        point = self.regularize(point)

        angle = np.linalg.norm(point)
        if np.isclose(angle, 0):
            coef_1 = 1 - angle ** 2 / 12
            coef_2 = 1 / 12 + angle ** 2 / 720
        elif np.isclose(angle, np.pi):
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
        return tangent_vec

    def group_log_from_identity(self, point):
        """
        Compute the group logarithm of point point.
        """
        point = self.regularize(point)
        return point

    def group_exp(self, tangent_vec, base_point=None):
        """
        Compute the group exponential of vector tangent_vector.
        """
        base_point = self.regularize(base_point)

        point = super(SpecialOrthogonalGroup, self).group_exp(
                                     tangent_vec=tangent_vec,
                                     base_point=base_point)
        point = self.regularize(point)
        return point

    def group_log(self, point, base_point=None):
        """
        Compute the group logarithm of point point.
        """
        point = self.regularize(point)
        base_point = self.regularize(base_point)

        tangent_vec = super(SpecialOrthogonalGroup, self).group_log(
                                    point=point,
                                    base_point=base_point)
        return tangent_vec

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
