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
    if mat.ndim == 2:
        mat = np.expand_dims(mat, axis=0)
    n_mats = mat.shape[0]
    assert mat.shape == (n_mats, 3, 3)

    mat_unitary_u, diag_s, mat_unitary_v = np.linalg.svd(mat)
    rot_mat = np.matmul(mat_unitary_u, mat_unitary_v)

    mask = np.where(np.linalg.det(rot_mat) < 0)

    if np.any(mask):
        diag_s[mask] = np.array([1, 1, -1])

        mat_diag_s = np.zeros_like(mat_unitary_u)
        for i in range(n_mats):
            mat_diag_s[i] = np.diag(diag_s[i])

        rot_mat[mask] = np.matmul(np.matmul(mat_unitary_u, mat_diag_s),
                                  mat_unitary_v)
    assert rot_mat.ndim == 3
    return rot_mat


def skew_matrix_from_vector(vec):
    """
    Compute the skew-symmetric matrix,
    known as the cross-product of a vector,
    associated to the vector vec.

    :param vec: 3d vector
    :return skew_mat: 3x3 skew-symmetric matrix
    """
    if vec.ndim == 1:
        vec = np.expand_dims(vec, axis=0)
    n_vecs = vec.shape[0]

    skew_mat = np.zeros((n_vecs, vec.shape[1], vec.shape[1]))
    for i in range(n_vecs):
        skew_mat[i] = np.cross(np.eye(3), vec[i])

    assert skew_mat.ndim == 3
    return skew_mat


def vector_from_skew_matrix(skew_mat):
    """
    Compute the skew-symmetric matrix,
    known as the cross-product of a vector,
    associated to the vector vec.

    :param skew_mat: 3x3 skew-symmetric matrix
    :return vec: 3d vector
    """
    if skew_mat.ndim == 2:
        skew_mat = np.expand_dims(skew_mat, axis=0)
    n_skew_mats = skew_mat.shape[0]

    assert skew_mat.shape == (n_skew_mats, 3, 3)

    vec = np.zeros((n_skew_mats, 3))
    vec[:] = skew_mat[:, (2, 0, 1), (1, 2, 0)]

    assert vec.ndim == 2
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
        if rot_vec.ndim == 1:
            rot_vec = np.expand_dims(rot_vec, axis=0)

        assert rot_vec.ndim == 2

        return rot_vec.shape[1] == self.dimension

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
        if rot_vec.ndim == 1:
            rot_vec = np.expand_dims(rot_vec, axis=0)

        angle = np.linalg.norm(rot_vec)
        regularized_rot_vec = rot_vec

        if angle != 0:
            k = np.floor(angle / (2 * np.pi) + .5)
            regularized_rot_vec = (1. - 2. * np.pi * k / angle) * rot_vec

        assert regularized_rot_vec.ndim == 2
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
        if rot_mat.ndim == 2:
            rot_mat = np.expand_dims(rot_mat, axis=0)
        n_rot_mats = rot_mat.shape[0]

        assert rot_mat.shape == (n_rot_mats, self.n, self.n)

        rot_mat = closest_rotation_matrix(rot_mat)

        trace = np.trace(rot_mat, axis1=1, axis2=2)
        if trace.ndim == 1:
            trace = np.expand_dims(trace, axis=1)
        assert trace.shape == (n_rot_mats, 1), trace.shape

        cos_angle = .5 * (trace - 1)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        rot_mat_transpose = np.transpose(rot_mat, axes=(0, 2, 1))
        rot_vec = vector_from_skew_matrix(rot_mat - rot_mat_transpose)

        mask_0 = np.isclose(angle, 0)
        mask_0 = np.squeeze(mask_0, axis=1)
        if np.any(mask_0):
            rot_vec[mask_0] = (rot_vec[mask_0]
                               * (.5 - (trace[mask_0] - 3.) / 12.))

        mask_pi = np.isclose(angle, np.pi)
        mask_pi = np.squeeze(mask_pi, axis=1)
        if np.any(mask_pi):
            # choose the largest diagonal element
            # to avoid a square root of a negative number
            a = np.argmax(np.diagonal(rot_mat, axis1=1, axis2=2))
            b = np.mod(a + 1, 3)
            c = np.mod(a + 2, 3)

            # compute the axis vector
            sq_root = np.sqrt((rot_mat[:, a, a]
                              - rot_mat[:, b, b] - rot_mat[:, c, c] + 1.))
            rot_vec_pi = np.zeros((n_rot_mats, self.dimension))
            rot_vec_pi[:, a] = sq_root / 2.
            rot_vec_pi[:, b] = ((rot_mat[:, b, a] + rot_mat[:, a, b])
                                / (2. * sq_root))
            rot_vec_pi[:, c] = ((rot_mat[:, c, a] + rot_mat[:, a, c])
                                / (2. * sq_root))

            rot_vec[mask_pi] = (angle[mask_pi] * rot_vec_pi[mask_pi]
                                / np.linalg.norm(rot_vec_pi[mask_pi]))

        mask_else = ~mask_0 & ~mask_pi
        if np.any(mask_else):
            rot_vec[mask_else] = (angle[mask_else]
                                  / (2. * np.sin(angle[mask_else]))
                                  * rot_vec[mask_else])

        return self.regularize(rot_vec)

    def matrix_from_rotation_vector(self, rot_vec):
        """
        Convert rotation vector to rotation matrix.

        :param rot_vec: 3d rotation vector
        :returns rot_mat: 3x3 rotation matrix

        """
        assert self.belongs(rot_vec)
        rot_vec = self.regularize(rot_vec)
        n_rot_vecs = rot_vec.shape[0]

        angle = np.linalg.norm(rot_vec, axis=1)
        angle = np.expand_dims(angle, axis=1)
        assert angle.shape == (n_rot_vecs, 1)
        skew_rot_vec = skew_matrix_from_vector(rot_vec)

        coef_1 = np.zeros([n_rot_vecs, 1])
        coef_2 = np.zeros([n_rot_vecs, 1])

        mask_0 = np.isclose(angle, 0)
        if np.any(mask_0):
            coef_1[mask_0] = 1 - (angle[mask_0] ** 2) / 6
            coef_2[mask_0] = 1 / 2 - angle[mask_0] ** 2
        if np.any(~mask_0):
            coef_1[~mask_0] = np.sin(angle[~mask_0]) / angle[~mask_0]
            coef_2[~mask_0] = ((1 - np.cos(angle[~mask_0]))
                               / (angle[~mask_0] ** 2))
        assert coef_1.shape == (n_rot_vecs, 1)
        assert coef_1.shape == (n_rot_vecs, 1)

        term_1 = np.zeros((n_rot_vecs, self.n, self.n))
        term_2 = np.zeros_like(term_1)

        for i in range(n_rot_vecs):
            term_1[i] = np.eye(self.dimension) + coef_1[i] * skew_rot_vec[i]
            term_2[i] = coef_2[i] * np.matmul(skew_rot_vec[i], skew_rot_vec[i])
        rot_mat = term_1 + term_2

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

        rot_vec_prod = self.regularize(rot_vec_prod)
        return rot_vec_prod

    def inverse(self, rot_vec):
        """
        Inverse of a rotation.
        """
        rot_vec = self.regularize(rot_vec)
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
        n_points = point.shape[0]

        angle = np.linalg.norm(point, axis=1)
        angle = np.expand_dims(angle, axis=1)

        coef_1 = np.zeros([n_points, 1])
        coef_2 = np.zeros([n_points, 1])

        mask_0 = np.isclose(angle, 0)
        mask_0 = np.squeeze(mask_0, axis=1)
        if np.any(mask_0):
            coef_1[mask_0] = 1 - angle[mask_0] ** 2 / 12
            coef_2[mask_0] = 1 / 12 + angle[mask_0] ** 2 / 720

        mask_pi = np.isclose(angle, np.pi)
        mask_pi = np.squeeze(mask_pi, axis=1)
        if np.any(mask_pi):
            coef_1[mask_pi] = angle[mask_pi] * (np.pi - angle[mask_pi]) / 4
            coef_2[mask_pi] = (1 - coef_1[mask_pi]) / angle[mask_pi] ** 2

        mask_else = ~mask_0 & ~mask_pi
        if np.any(mask_else):
            coef_1[mask_else] = ((angle[mask_else] / 2)
                                 / np.tan(angle[mask_else] / 2))
            coef_2[mask_else] = (1 - coef_1[mask_else]) / angle[mask_else] ** 2

        jacobian = np.zeros((n_points, self.dimension, self.dimension))

        for i in range(n_points):
            if left_or_right == 'left':
                jacobian[i] = (coef_1[i] * np.identity(self.dimension)
                               + coef_2[i] * np.outer(point[i], point[i])
                               + skew_matrix_from_vector(point[i]) / 2)

            else:
                jacobian[i] = (coef_1[i] * np.identity(self.dimension)
                               + coef_2[i] * np.outer(point[i], point[i])
                               - skew_matrix_from_vector(point[i]) / 2)

        assert jacobian.ndim == 3
        return jacobian

    def random_uniform(self, n_samples=1):
        """
        Sample a 3d rotation vector uniformly, w.r.t.
        the bi-invariant metric, by sampling in the
        hypercube of side [-1, 1] on the tangent space.
        """
        random_rot_vec = np.random.rand(n_samples, self.dimension) * 2 - 1
        random_rot_vec = self.regularize(random_rot_vec)
        return random_rot_vec

    def group_exp_from_identity(self, tangent_vec):
        """
        Compute the group exponential of vector tangent_vector.
        """
        if tangent_vec.ndim == 1:
            tangent_vec = np.expand_dims(tangent_vec, axis=0)
        assert tangent_vec.ndim == 2
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
        if tangent_vec.ndim == 1:
            tangent_vec = np.expand_dims(tangent_vec, axis=0)
        assert tangent_vec.ndim == 2

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
        assert tangent_vec.ndim == 2
        return tangent_vec

    def group_exponential_barycenter(self, points, weights=None):
        """
        Group exponential barycenter is the Frechet mean
        of the bi-invariant metric.
        """
        n_points = points.shape[0]
        assert n_points > 0

        if weights is None:
            weights = np.ones((n_points, 1))

        n_weights = weights.shape[0]
        assert n_points == n_weights

        barycenter = self.bi_invariant_metric.mean(points, weights)
        assert barycenter.ndim == 2
        return barycenter
