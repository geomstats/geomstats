"""Computations on the Lie group of 3D rotations."""

# TODO(nina): Rename modules to make imports cleaner?
# TODO(nina): make code robust to different types and input structures
from geomstats.embedded_manifold import EmbeddedManifold
from geomstats.general_linear_group import GeneralLinearGroup
from geomstats.lie_group import LieGroup
import geomstats.backend as gs
import geomstats.spd_matrices_space as spd_matrices_space

ATOL = 1e-5


def closest_rotation_matrix(mat):
    """
    Compute the closest - in terms of
    the Frobenius norm - rotation matrix
    of a given matrix mat.

    :param mat: matrix
    :returns rot_mat: rotation matrix.
    """
    mat = gs.to_ndarray(mat, to_ndim=3)

    n_mats, mat_dim_1, mat_dim_2 = mat.shape
    assert mat_dim_1 == mat_dim_2

    if mat_dim_1 == 3:
        mat_unitary_u, diag_s, mat_unitary_v = gs.linalg.svd(mat)
        rot_mat = gs.matmul(mat_unitary_u, mat_unitary_v)

        mask = gs.where(gs.linalg.det(rot_mat) < 0)
        new_mat_diag_s = gs.tile(gs.diag([1, 1, -1]), len(mask))

        rot_mat[mask] = gs.matmul(gs.matmul(mat_unitary_u[mask],
                                            new_mat_diag_s),
                                  mat_unitary_v[mask])
    else:
        aux_mat = gs.matmul(gs.transpose(mat, axes=(0, 2, 1)), mat)

        inv_sqrt_mat = gs.zeros_like(mat)
        for i in range(n_mats):
            sym_mat = aux_mat[i]

            assert spd_matrices_space.is_symmetric(sym_mat)
            inv_sqrt_mat[i] = gs.linalg.inv(spd_matrices_space.sqrtm(sym_mat))
        rot_mat = gs.matmul(mat, inv_sqrt_mat)

    assert rot_mat.ndim == 3
    return rot_mat


def skew_matrix_from_vector(vec):
    """
    In 3D, compute the skew-symmetric matrix,
    known as the cross-product of a vector,
    associated to the vector vec.

    In nD, fill a skew-symmetric matrix with
    the values of the vector.

    :param vec: vector
    :return skew_mat: skew-symmetric matrix
    """
    vec = gs.to_ndarray(vec, to_ndim=2)
    n_vecs, vec_dim = vec.shape

    mat_dim = int((1 + gs.sqrt(1 + 8 * vec_dim)) / 2)
    skew_mat = gs.zeros((n_vecs,) + (mat_dim,) * 2)

    if vec_dim == 3:
        for i in range(n_vecs):
            skew_mat[i] = gs.cross(gs.eye(vec_dim), vec[i])
    else:
        upper_triangle_indices = gs.triu_indices(mat_dim, k=1)
        for i in range(n_vecs):
            skew_mat[i][upper_triangle_indices] = vec[i]
            skew_mat[i] = skew_mat[i] - skew_mat[i].transpose()

    assert skew_mat.ndim == 3
    return skew_mat


def vector_from_skew_matrix(skew_mat):
    """
    In 3D, compute the vector defining the cross product
    associated to the skew-symmetric matrix skew mat.

    In nD, fill a vector by reading the values
    of the upper triangle of skew_mat.

    :param skew_mat: skew-symmetric matrix
    :return vec: vector
    """
    skew_mat = gs.to_ndarray(skew_mat, to_ndim=3)
    n_skew_mats, mat_dim_1, mat_dim_2 = skew_mat.shape

    assert mat_dim_1 == mat_dim_2

    vec_dim = int(mat_dim_1 * (mat_dim_1 - 1) / 2)
    vec = gs.zeros((n_skew_mats, vec_dim))

    if vec_dim == 3:
        vec[:] = skew_mat[:, (2, 0, 1), (1, 2, 0)]
    else:
        idx = 0
        for j in range(mat_dim_1):
            for i in range(j):
                vec[:, idx] = skew_mat[:, i, j]
                idx += 1

    assert vec.ndim == 2
    return vec


class SpecialOrthogonalGroup(LieGroup, EmbeddedManifold):

    def __init__(self, n):
        assert isinstance(n, int) and n > 1

        self.n = n
        self.dimension = int((n * (n - 1)) / 2)
        LieGroup.__init__(self,
                          dimension=self.dimension,
                          identity=gs.zeros(self.dimension))
        EmbeddedManifold.__init__(self,
                                  dimension=self.dimension,
                                  embedding_manifold=GeneralLinearGroup(n=n))
        self.bi_invariant_metric = self.left_canonical_metric
        self.point_representation = 'vector' if n == 3 else 'matrix'

    def belongs(self, rot_vec):
        """
        Check that a vector belongs to the
        special orthogonal group.
        """
        rot_vec = gs.to_ndarray(rot_vec, to_ndim=2)
        _, vec_dim = rot_vec.shape
        return vec_dim == self.dimension

    def regularize(self, rot_vec):
        """
        In 3D, regularize the norm of the rotation vector,
        to be between 0 and pi, following the axis-angle
        representation's convention.

        If the angle angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        :param rot_vec: 3d vector
        :returns self.regularized_rot_vec: 3d vector with: 0 < norm < pi
        """
        rot_vec = gs.to_ndarray(rot_vec, to_ndim=2)
        assert self.belongs(rot_vec)
        n_rot_vecs, vec_dim = rot_vec.shape

        if vec_dim == 3:
            angle = gs.linalg.norm(rot_vec, axis=1)
            regularized_rot_vec = rot_vec.astype('float64')
            mask_not_0 = angle != 0

            k = gs.floor(angle / (2 * gs.pi) + .5)
            norms_ratio = gs.zeros_like(angle).astype('float64')
            norms_ratio[mask_not_0] = (
                  1. - 2. * gs.pi * k[mask_not_0] / angle[mask_not_0])
            norms_ratio[angle == 0] = 1
            for i in range(n_rot_vecs):
                regularized_rot_vec[i, :] = norms_ratio[i] * rot_vec[i]
        else:
            # TODO(nina): regularization needed in nD?
            regularized_rot_vec = rot_vec

        assert regularized_rot_vec.ndim == 2
        return regularized_rot_vec

    def regularize_tangent_vec_at_identity(self, tangent_vec, metric=None):
        """
        In 3D, regularize a tangent_vector by getting its norm at the identity,
        determined by the metric, to be less than pi,
        following the regularization convention.
        """
        assert self.point_representation in ('vector', 'matrix')

        if self.point_representation is 'vector':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
            _, vec_dim = tangent_vec.shape

            if metric is None:
                metric = self.left_canonical_metric
            tangent_vec_metric_norm = metric.norm(tangent_vec)
            tangent_vec_canonical_norm = gs.linalg.norm(tangent_vec, axis=1)
            if tangent_vec_canonical_norm.ndim == 1:
                tangent_vec_canonical_norm = gs.expand_dims(
                                         tangent_vec_canonical_norm, axis=1)

            mask_norm_0 = gs.isclose(tangent_vec_metric_norm, 0)
            mask_canonical_norm_0 = gs.isclose(tangent_vec_canonical_norm, 0)

            mask_0 = mask_norm_0 | mask_canonical_norm_0
            mask_else = ~mask_0

            mask_0 = gs.squeeze(mask_0, axis=1)
            mask_else = gs.squeeze(mask_else, axis=1)

            coef = gs.empty_like(tangent_vec_metric_norm)
            regularized_vec = tangent_vec

            regularized_vec[mask_0] = tangent_vec[mask_0]

            coef[mask_else] = (tangent_vec_metric_norm[mask_else]
                               / tangent_vec_canonical_norm[mask_else])
            regularized_vec[mask_else] = self.regularize(
                    coef[mask_else] * tangent_vec[mask_else])
            regularized_vec[mask_else] = (regularized_vec[mask_else]
                                          / coef[mask_else])
        else:
            # TODO(nina): regularization needed in nD?
            regularized_vec = tangent_vec

        return regularized_vec

    def regularize_tangent_vec(self, tangent_vec, base_point, metric=None):
        """
        Regularize a tangent_vector by getting its norm at the identity,
        determined by the metric,
        to be less than pi,
        following the regularization convention
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        _, vec_dim = tangent_vec.shape
        if vec_dim == 3:
            if metric is None:
                metric = self.left_canonical_metric
            base_point = self.regularize(base_point)

            jacobian = self.jacobian_translation(
                                          point=base_point,
                                          left_or_right=metric.left_or_right)
            inv_jacobian = gs.linalg.inv(jacobian)
            tangent_vec_at_id = gs.dot(
                    tangent_vec,
                    gs.transpose(inv_jacobian, axes=(0, 2, 1)))
            tangent_vec_at_id = gs.squeeze(tangent_vec_at_id, axis=1)

            tangent_vec_at_id = self.regularize_tangent_vec_at_identity(
                                          tangent_vec_at_id,
                                          metric)

            regularized_tangent_vec = gs.dot(tangent_vec_at_id,
                                             gs.transpose(jacobian,
                                                          axes=(0, 2, 1)))
            regularized_tangent_vec = gs.squeeze(regularized_tangent_vec,
                                                 axis=1)
        else:
            # TODO(nina): is regularization needed in nD?
            regularized_tangent_vec = tangent_vec
        return regularized_tangent_vec

    def rotation_vector_from_matrix(self, rot_mat):
        """
        In 3D, convert rotation matrix to rotation vector
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

        In nD, the rotation vector stores the n(n-1)/2 values of the
        skew-symmetric matrix representing the rotation.

        :param rot_mat: rotation matrix
        :return rot_vec: rotation vector
        """
        rot_mat = gs.to_ndarray(rot_mat, to_ndim=3)
        n_rot_mats, mat_dim_1, mat_dim_2 = rot_mat.shape
        assert mat_dim_1 == mat_dim_2 == self.n

        rot_mat = closest_rotation_matrix(rot_mat)

        if self.n == 3:
            trace = gs.trace(rot_mat, axis1=1, axis2=2)
            trace = gs.to_ndarray(trace, to_ndim=2, axis=1)
            assert trace.shape == (n_rot_mats, 1), trace.shape

            cos_angle = .5 * (trace - 1)
            cos_angle = gs.clip(cos_angle, -1, 1)
            angle = gs.arccos(cos_angle)

            rot_mat_transpose = gs.transpose(rot_mat, axes=(0, 2, 1))
            rot_vec = vector_from_skew_matrix(rot_mat - rot_mat_transpose)

            mask_0 = gs.isclose(angle, 0)
            mask_0 = gs.squeeze(mask_0, axis=1)
            rot_vec[mask_0] = (rot_vec[mask_0]
                               * (.5 - (trace[mask_0] - 3.) / 12.))

            mask_pi = gs.isclose(angle, gs.pi)
            mask_pi = gs.squeeze(mask_pi, axis=1)

            # choose the largest diagonal element
            # to avoid a square root of a negative number
            a = 0
            if gs.any(mask_pi):
                a = gs.argmax(gs.diagonal(rot_mat[mask_pi], axis1=1, axis2=2))
            b = gs.mod(a + 1, 3)
            c = gs.mod(a + 2, 3)

            # compute the axis vector
            sq_root = gs.sqrt((rot_mat[mask_pi, a, a]
                               - rot_mat[mask_pi, b, b]
                               - rot_mat[mask_pi, c, c] + 1.))
            rot_vec_pi = gs.zeros((sum(mask_pi), self.dimension))
            rot_vec_pi[:, a] = sq_root / 2.
            rot_vec_pi[:, b] = ((rot_mat[mask_pi, b, a]
                                 + rot_mat[mask_pi, a, b])
                                / (2. * sq_root))
            rot_vec_pi[:, c] = ((rot_mat[mask_pi, c, a]
                                + rot_mat[mask_pi, a, c])
                                / (2. * sq_root))

            rot_vec[mask_pi] = (angle[mask_pi] * rot_vec_pi
                                / gs.linalg.norm(rot_vec_pi))

            mask_else = ~mask_0 & ~mask_pi
            rot_vec[mask_else] = (angle[mask_else]
                                  / (2. * gs.sin(angle[mask_else]))
                                  * rot_vec[mask_else])
        else:
            skew_mat = self.embedding_manifold.group_log_from_identity(rot_mat)
            rot_vec = vector_from_skew_matrix(skew_mat)

        return self.regularize(rot_vec)

    def matrix_from_rotation_vector(self, rot_vec):
        """
        Convert rotation vector to rotation matrix.

        :param rot_vec: rotation vector
        :returns rot_mat: rotation matrix

        """
        assert self.belongs(rot_vec)
        rot_vec = self.regularize(rot_vec)
        n_rot_vecs, _ = rot_vec.shape

        if self.n == 3:
            angle = gs.linalg.norm(rot_vec, axis=1)
            angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

            skew_rot_vec = skew_matrix_from_vector(rot_vec)

            coef_1 = gs.zeros_like(angle)
            coef_2 = gs.zeros_like(angle)

            mask_0 = gs.isclose(angle, 0)
            coef_1[mask_0] = 1 - (angle[mask_0] ** 2) / 6
            coef_2[mask_0] = 1 / 2 - angle[mask_0] ** 2

            coef_1[~mask_0] = gs.sin(angle[~mask_0]) / angle[~mask_0]
            coef_2[~mask_0] = ((1 - gs.cos(angle[~mask_0]))
                               / (angle[~mask_0] ** 2))

            term_1 = gs.zeros((n_rot_vecs,) + (self.n,) * 2)
            term_2 = gs.zeros_like(term_1)

            for i in range(n_rot_vecs):
                term_1[i] = (gs.eye(self.dimension)
                             + coef_1[i] * skew_rot_vec[i])
                term_2[i] = (coef_2[i]
                             * gs.matmul(skew_rot_vec[i], skew_rot_vec[i]))
            rot_mat = term_1 + term_2

            rot_mat = closest_rotation_matrix(rot_mat)

        else:
            skew_mat = skew_matrix_from_vector(rot_vec)
            rot_mat = self.embedding_manifold.group_exp_from_identity(skew_mat)

        return rot_mat

    def quaternion_from_matrix(self, rot_mat):
        """
        Convert a rotation matrix into a unit quaternion.
        """
        assert self.n == 3, ('The quaternion representation does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        rot_mat = gs.to_ndarray(rot_mat, to_ndim=3)

        rot_vec = self.rotation_vector_from_matrix(rot_mat)
        quaternion = self.quaternion_from_rotation_vector(rot_vec)

        assert quaternion.ndim == 2
        return quaternion

    def quaternion_from_rotation_vector(self, rot_vec):
        """
        Convert a rotation vector into a unit quaternion.
        """
        assert self.n == 3, ('The quaternion representation does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        rot_vec = self.regularize(rot_vec)
        n_rot_vecs, _ = rot_vec.shape

        angle = gs.linalg.norm(rot_vec, axis=1)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

        rotation_axis = gs.zeros_like(rot_vec)

        mask_0 = gs.isclose(angle, 0)
        mask_0 = gs.squeeze(mask_0, axis=1)
        mask_not_0 = ~mask_0
        rotation_axis[mask_not_0] = rot_vec[mask_not_0] / angle[mask_not_0]

        n_quaternions, _ = rot_vec.shape
        quaternion = gs.zeros((n_quaternions, 4))
        quaternion[:, :1] = gs.cos(angle / 2)
        quaternion[:, 1:] = gs.sin(angle / 2) * rotation_axis[:]

        return quaternion

    def rotation_vector_from_quaternion(self, quaternion):
        """
        Convert a unit quaternion into a rotation vector.
        """
        assert self.n == 3, ('The quaternion representation does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        quaternion = gs.to_ndarray(quaternion, to_ndim=2)
        n_quaternions, _ = quaternion.shape

        cos_half_angle = quaternion[:, 0]
        cos_half_angle = gs.clip(cos_half_angle, -1, 1)
        half_angle = gs.arccos(cos_half_angle)

        half_angle = gs.to_ndarray(half_angle, to_ndim=2, axis=1)
        assert half_angle.shape == (n_quaternions, 1)

        rot_vec = gs.zeros_like(quaternion[:, 1:])

        mask_0 = gs.isclose(half_angle, 0)
        mask_0 = gs.squeeze(mask_0, axis=1)
        mask_not_0 = ~mask_0
        rotation_axis = (quaternion[mask_not_0, 1:]
                         / gs.sin(half_angle[mask_not_0]))
        rot_vec[mask_not_0] = (2 * half_angle[mask_not_0]
                               * rotation_axis)

        rot_vec = self.regularize(rot_vec)
        return rot_vec

    def matrix_from_quaternion(self, quaternion):
        """
        Convert a unit quaternion into a rotation vector.
        """
        assert self.n == 3, ('The quaternion representation does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        quaternion = gs.to_ndarray(quaternion, to_ndim=2)
        n_quaternions, _ = quaternion.shape

        a, b, c, d = gs.hsplit(quaternion, 4)

        rot_mat = gs.zeros((n_quaternions,) + (self.n,) * 2)

        for i in range(n_quaternions):
            # TODO(nina): vectorize by applying the composition of
            # quaternions to the identity matrix
            column_1 = [a[i] ** 2 + b[i] ** 2 - c[i] ** 2 - d[i] ** 2,
                        2 * b[i] * c[i] - 2 * a[i] * d[i],
                        2 * b[i] * d[i] + 2 * a[i] * c[i]]

            column_2 = [2 * b[i] * c[i] + 2 * a[i] * d[i],
                        a[i] ** 2 - b[i] ** 2 + c[i] ** 2 - d[i] ** 2,
                        2 * c[i] * d[i] - 2 * a[i] * b[i]]

            column_3 = [2 * b[i] * d[i] - 2 * a[i] * c[i],
                        2 * c[i] * d[i] + 2 * a[i] * b[i],
                        a[i] ** 2 - b[i] ** 2 - c[i] ** 2 + d[i] ** 2]

            rot_mat[i] = gs.hstack([column_1, column_2, column_3]).transpose()

        assert rot_mat.ndim == 3
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

        rot_mat_prod = gs.einsum('ijk,ikl->ijl', rot_mat_1, rot_mat_2)
        rot_vec_prod = self.rotation_vector_from_matrix(rot_mat_prod)

        rot_vec_prod = self.regularize(rot_vec_prod)
        return rot_vec_prod

    def inverse(self, rot_vec):
        """
        Inverse of a rotation.
        """
        if self.n == 3:
            inv_rot_vec = -self.regularize(rot_vec)
        else:
            rot_mat = self.matrix_from_rotation_vector(rot_vec)
            inv_rot_mat = gs.linalg.inv(rot_mat)
            inv_rot_vec = self.rotation_vector_from_matrix(inv_rot_mat)
        return inv_rot_vec

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

        if self.n == 3:
            point = self.regularize(point)
            n_points, _ = point.shape

            angle = gs.linalg.norm(point, axis=1)
            angle = gs.expand_dims(angle, axis=1)

            coef_1 = gs.zeros([n_points, 1])
            coef_2 = gs.zeros([n_points, 1])

            mask_0 = gs.isclose(angle, 0)
            mask_0 = gs.squeeze(mask_0, axis=1)
            coef_1[mask_0] = (1 - angle[mask_0] ** 2 / 12
                              - angle[mask_0] ** 4 / 720
                              - angle[mask_0] ** 6 / 30240)
            coef_2[mask_0] = (1 / 12 + angle[mask_0] ** 2 / 720
                              + angle[mask_0] ** 4 / 30240
                              + angle[mask_0] ** 6 / 1209600)

            mask_pi = gs.isclose(angle, gs.pi)
            mask_pi = gs.squeeze(mask_pi, axis=1)
            delta_angle = angle[mask_pi] - gs.pi
            coef_1[mask_pi] = (- gs.pi * delta_angle / 4
                               - delta_angle ** 2 / 4
                               - gs.pi * delta_angle ** 3 / 48
                               - delta_angle ** 4 / 48
                               - gs.pi * delta_angle ** 5 / 480
                               - delta_angle ** 6 / 480)
            coef_2[mask_pi] = (1 - coef_1[mask_pi]) / angle[mask_pi] ** 2

            mask_else = ~mask_0 & ~mask_pi
            coef_1[mask_else] = ((angle[mask_else] / 2)
                                 / gs.tan(angle[mask_else] / 2))
            coef_2[mask_else] = (1 - coef_1[mask_else]) / angle[mask_else] ** 2

            jacobian = gs.zeros((n_points, self.dimension, self.dimension))

            for i in range(n_points):
                if left_or_right == 'left':
                    jacobian[i] = (coef_1[i] * gs.identity(self.dimension)
                                   + coef_2[i] * gs.outer(point[i], point[i])
                                   + skew_matrix_from_vector(point[i]) / 2)

                else:
                    jacobian[i] = (coef_1[i] * gs.identity(self.dimension)
                                   + coef_2[i] * gs.outer(point[i], point[i])
                                   - skew_matrix_from_vector(point[i]) / 2)

        else:
            if left_or_right == 'right':
                raise NotImplementedError(
                    'The jacobian of the right translation'
                    ' is not implemented.')
            jacobian = self.matrix_from_rotation_vector(point)

        assert jacobian.ndim == 3

        return jacobian

    def random_uniform(self, n_samples=1):
        """
        Sample a 3d rotation vector uniformly, w.r.t.
        the bi-invariant metric, by sampling in the
        hypercube of side [-1, 1] on the tangent space.
        """
        random_rot_vec = gs.random.rand(n_samples, self.dimension) * 2 - 1
        random_rot_vec = self.regularize(random_rot_vec)
        return random_rot_vec

    def group_exp_from_identity(self, tangent_vec):
        """
        Compute the group exponential of vector tangent_vector.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
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
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)

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
            weights = gs.ones((n_points, 1))

        n_weights = weights.shape[0]
        assert n_points == n_weights

        barycenter = self.bi_invariant_metric.mean(points, weights)

        barycenter = gs.to_ndarray(barycenter, to_ndim=2)
        assert barycenter.ndim == 2, barycenter.ndim
        return barycenter
