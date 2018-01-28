"""Computations on the Lie group of 3D rotations."""

import numpy as np

from geomstats.invariant_metric import InvariantMetric
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


class BiinvariantMetric(InvariantMetric):
    def __init__(self, group):
        super(BiinvariantMetric, self).__init__(group, np.eye(3))

    def left_exp_from_identity(self, tangent_vec):
        """
        Compute the *left* Riemannian exponential from the identity of the
        Lie group of tangent vector tangent_vec.

        The left Riemannian exponential has a special role since the
        left Riemannian exponential of the canonical metric parameterizes
        the points.

        Note: In the case where the method is called by a right-invariant
        metric, it used the left-invariant metric associated to the same
        inner-product at the identity.
        """
        exp = np.dot(self.inner_product_mat_at_identity, tangent_vec)

        exp = self.lie_group.regularize(exp)
        return exp

    def exp_from_identity(self, tangent_vec):
        """
        Compute the Riemannian exponential from the identity of the
        Lie group of tangent vector tangent_vec.
        """
        if self.left_or_right == 'left':
            exp = self.left_exp_from_identity(tangent_vec)

        else:
            opp_left_exp = self.left_exp_from_identity(-tangent_vec)

            exp = self.lie_group.inverse(opp_left_exp)

        exp = self.lie_group.regularize(exp)
        return exp

    def exp(self, tangent_vec, base_point):
        """
        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec.
        """
        base_point = self.lie_group.regularize(base_point)

        jacobian = self.lie_group.jacobian_translation(
                                 base_point,
                                 left_or_right=self.left_or_right)
        inv_jacobian = np.linalg.inv(jacobian)

        tangent_vec_translated_to_id = np.dot(inv_jacobian, tangent_vec)

        exp_from_id = self.exp_from_identity(
                               tangent_vec_translated_to_id)

        if self.left_or_right == 'left':
            exp = self.lie_group.compose(base_point, exp_from_id)

        else:
            exp = self.lie_group.compose(exp_from_id, base_point)

        exp = self.lie_group.regularize(exp)
        return exp

    def left_log_from_identity(self, point):
        """
        Compute the *left* Riemannian logarithm from the identity of the
        Lie group of tangent vector tangent_vec.

        The left Riemannian logarithm has a special role since the
        left Riemannian logarithm of the canonical metric parameterizes
        the points.
        """
        point = self.lie_group.regularize(point)

        inner_prod_mat = self.inner_product_mat_at_identity
        inv_inner_prod_mat = np.linalg.inv(inner_prod_mat)

        log = np.dot(inv_inner_prod_mat, point)

        return log

    def log_from_identity(self, point):
        """
        Compute the Riemannian logarithm of point at point base_point
        of point for the invariant metric from the identity.
        """
        point = self.lie_group.regularize(point)
        if self.left_or_right == 'left':
            log = self.left_log_from_identity(point)

        else:
            inv_point = self.lie_group.inverse(point)
            left_log = self.left_log_from_identity(inv_point)
            log = - left_log

        return log

    def log(self, point, base_point):
        """
        Compute the Riemannian logarithm of point at point base_point
        of point for the invariant metric.
        """
        base_point = self.lie_group.regularize(base_point)
        point = self.lie_group.regularize(point)

        if self.left_or_right == 'left':
            point_near_id = self.lie_group.compose(
                                   self.lie_group.inverse(base_point),
                                   point)

        else:
            point_near_id = self.lie_group.compose(
                                   point,
                                   self.lie_group.inverse(base_point))

        log_from_id = self.log_from_identity(point_near_id)

        jacobian = self.lie_group.jacobian_translation(
                                       base_point,
                                       left_or_right=self.left_or_right)
        log = np.dot(jacobian, log_from_id)

        return log


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
        self.bi_invariant_metric = BiinvariantMetric(self)

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
        rot_vec = rot_vec.astype(dtype=np.float64)
        angle = np.linalg.norm(rot_vec)
        regularized_rot_vec = rot_vec

        if angle != 0:
            k = np.floor(angle / (2 * np.pi) + .5)
            regularized_rot_vec = (1. - 2. * np.pi * k / angle) * rot_vec

        return regularized_rot_vec

    def rotation_vector_from_matrix(self, rot_mat, epsilon=EPSILON):
        """
        Convert rotation matrix to rotation vector
        (axis-angle representation).

        :param rot_mat: 3x3 rotation matrix
        :returns rot_vec: 3d rotation vector
        """
        assert rot_mat.shape == (3, 3)

        rot_mat = closest_rotation_matrix(rot_mat)

        # t is the sum of the eigenvalues of the rot_mat.
        # The eigenvalues are:
        # 1, cos(theta) + i sin(theta), cos(theta) - i sin(theta)
        # trace = 1 + 2 cos(theta), -1 <= trace <= 3
        trace = np.trace(rot_mat, dtype=np.float64)
        cos_angle = .5 * (trace - 1)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle, dtype=np.float64)

        rot_vec = vector_from_skew_matrix(rot_mat - rot_mat.transpose())

        # -- angle is not close to 0 or pi
        if np.sin(angle) > epsilon:
            rot_vec = angle / (2. * np.sin(angle)) * rot_vec

        # -- Edge case: angle is close to 0
        elif trace - 1. > 0.:
            rot_vec = (.5 - (trace - 3.) / 12.) * rot_vec

        # -- Edge case: angle is close to pi
        else:
            # r = angle * v / |v|, where (w, v) is a unit quaternion.
            # This formulation is derived by going from rotation matrix to unit
            # quaternion to axis-angle

            # choose the largest diagonal element
            # to avoid a square root of a negative number
            a = np.argmax(np.diag(rot_mat))
            b = np.mod(a + 1, 3)
            c = np.mod(a + 2, 3)

            # compute the axis vector
            s = np.sqrt(rot_mat[a, a] - rot_mat[b, b] - rot_mat[c, c] + 1.)
            rot_vec = np.zeros(3)
            rot_vec[a] = s / 2.
            rot_vec[b] = (rot_mat[b, a] + rot_mat[a, b]) / (2. * s)
            rot_vec[c] = (rot_mat[c, a] + rot_mat[a, c]) / (2. * s)

            rot_vec = angle * rot_vec / np.linalg.norm(rot_vec)

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
                             left_or_right='left', epsilon=EPSILON):
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
        # tangent_vec = self.regularize(tangent_vec)
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
