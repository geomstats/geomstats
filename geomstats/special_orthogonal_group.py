"""
The special orthogonal group SO(n),
i.e. the Lie group of rotations in n dimensions.
"""

# TODO(nina): make code robust to different types and input structures
# TODO(nina): should the conversion functions be methods?
import geomstats.backend as gs
import geomstats.spd_matrices_space as spd_matrices_space

from geomstats.embedded_manifold import EmbeddedManifold
from geomstats.general_linear_group import GeneralLinearGroup
from geomstats.lie_group import LieGroup

ATOL = 1e-5

TAYLOR_COEFFS_1_AT_0 = [1., 0.,
                        - 1. / 12., 0.,
                        - 1. / 720., 0.,
                        - 1. / 30240., 0.]
TAYLOR_COEFFS_2_AT_0 = [1. / 12., 0.,
                        1. / 720., 0.,
                        1. / 30240., 0.,
                        1. / 1209600., 0.]
TAYLOR_COEFFS_1_AT_PI = [0., - gs.pi / 4.,
                         - 1. / 4., - gs.pi / 48.,
                         - 1. / 48., - gs.pi / 480.,
                         - 1. / 480.]


class SpecialOrthogonalGroup(LieGroup, EmbeddedManifold):
    """
    Class for the special orthogonal group SO(n),
    i.e. the Lie group of rotations.
    """

    def __init__(self, n, point_type=None):

        assert isinstance(n, int) and n > 1

        self.n = n
        self.dimension = int((n * (n - 1)) / 2)

        self.default_point_type = point_type
        if point_type is None:
            self.default_point_type = 'vector' if n == 3 else 'matrix'

        LieGroup.__init__(self,
                          dimension=self.dimension)
        EmbeddedManifold.__init__(self,
                                  dimension=self.dimension,
                                  embedding_manifold=GeneralLinearGroup(n=n))
        self.bi_invariant_metric = self.left_canonical_metric

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
        Evaluate if a point belongs to SO(n).
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            point = gs.to_ndarray(point, to_ndim=2)
            _, vec_dim = point.shape
            return vec_dim == self.dimension

        elif point_type == 'matrix':
            point = gs.to_ndarray(point, to_ndim=3)
            point_transpose = gs.transpose(point, axes=(0, 2, 1))
            point_inverse = gs.linalg.inv(point)

            mask = gs.isclose(point_inverse, point_transpose)
            mask = gs.all(mask, axis=(1, 2))

            return mask

    def regularize(self, point, point_type=None):
        """
        In 3D, regularize the norm of the rotation vector,
        to be between 0 and pi, following the axis-angle
        representation's convention.

        If the angle angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            point = gs.to_ndarray(point, to_ndim=2)
            assert self.belongs(point, point_type)
            n_points, _ = point.shape

            regularized_point = gs.copy(point)
            if self.n == 3:
                angle = gs.linalg.norm(regularized_point, axis=1)
                mask_0 = gs.isclose(angle, 0)
                mask_not_0 = ~mask_0

                mask_pi = gs.isclose(angle, gs.pi)

                k = gs.floor(angle / (2 * gs.pi) + .5)
                norms_ratio = gs.zeros_like(angle)
                norms_ratio[mask_not_0] = (
                      1. - 2. * gs.pi * k[mask_not_0] / angle[mask_not_0])
                norms_ratio[mask_0] = 1
                norms_ratio[mask_pi] = gs.pi / angle[mask_pi]
                for i in range(n_points):
                    regularized_point[i, :] = (norms_ratio[i]
                                               * regularized_point[i, :])
            else:
                # TODO(nina): regularization needed in nD?
                regularized_point = gs.copy(point)

            assert gs.ndim(regularized_point) == 2

        elif point_type == 'matrix':
            point = gs.to_ndarray(point, to_ndim=3)
            # TODO(nina): regularization for matrices?
            regularized_point = gs.copy(point)

        return regularized_point

    def regularize_tangent_vec_at_identity(
            self, tangent_vec, metric=None, point_type=None):
        """
        In 3D, regularize a tangent_vector by getting its norm at the identity,
        determined by the metric, to be less than pi.
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)

            if self.n == 3:
                if metric is None:
                    metric = self.left_canonical_metric
                tangent_vec_metric_norm = metric.norm(tangent_vec)
                tangent_vec_canonical_norm = gs.linalg.norm(
                                                  tangent_vec, axis=1)
                if gs.ndim(tangent_vec_canonical_norm) == 1:
                    tangent_vec_canonical_norm = gs.expand_dims(
                                   tangent_vec_canonical_norm, axis=1)

                mask_norm_0 = gs.isclose(tangent_vec_metric_norm, 0)
                mask_canonical_norm_0 = gs.isclose(
                    tangent_vec_canonical_norm, 0)

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

        elif point_type == 'matrix':
                # TODO(nina): regularization in terms
                # of skew-symmetric matrices?
                regularized_vec = tangent_vec

        return regularized_vec

    def regularize_tangent_vec(
            self, tangent_vec, base_point,
            metric=None, point_type=None):
        """
        In 3D, regularize a tangent_vector by getting the norm of its parallel
        transport to the identity, determined by the metric,
        to be less than pi.
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)

            if self.n == 3:
                if metric is None:
                    metric = self.left_canonical_metric
                base_point = self.regularize(base_point, point_type)

                jacobian = self.jacobian_translation(
                              point=base_point,
                              left_or_right=metric.left_or_right,
                              point_type=point_type)
                inv_jacobian = gs.linalg.inv(jacobian)
                tangent_vec_at_id = gs.dot(
                        tangent_vec,
                        gs.transpose(inv_jacobian, axes=(0, 2, 1)))
                tangent_vec_at_id = gs.squeeze(tangent_vec_at_id, axis=1)

                tangent_vec_at_id = self.regularize_tangent_vec_at_identity(
                                              tangent_vec_at_id,
                                              metric,
                                              point_type)

                regularized_tangent_vec = gs.dot(tangent_vec_at_id,
                                                 gs.transpose(jacobian,
                                                              axes=(0, 2, 1)))
                regularized_tangent_vec = gs.squeeze(regularized_tangent_vec,
                                                     axis=1)
            else:
                # TODO(nina): is regularization needed in nD?
                regularized_tangent_vec = tangent_vec

        elif point_type == 'matrix':
            # TODO(nina): regularization in terms
            # of skew-symmetric matrices?
            regularized_tangent_vec = tangent_vec

        return regularized_tangent_vec

    def projection(self, mat):
        """
        Project a matrix on SO(n), using the Frobenius norm.
        """
        # TODO(nina): projection when the point_type is not 'matrix'?
        mat = gs.to_ndarray(mat, to_ndim=3)

        n_mats, mat_dim_1, mat_dim_2 = mat.shape
        assert mat_dim_1 == mat_dim_2 == self.n

        if self.n == 3:
            mat_unitary_u, diag_s, mat_unitary_v = gs.linalg.svd(mat)
            rot_mat = gs.matmul(mat_unitary_u, mat_unitary_v)
            mask = gs.nonzero(gs.linalg.det(rot_mat) < 0)
            diag = gs.array([1, 1, -1])
            new_mat_diag_s = gs.tile(gs.diag(diag), len(mask))

            rot_mat[mask] = gs.matmul(gs.matmul(mat_unitary_u[mask],
                                                new_mat_diag_s),
                                      mat_unitary_v[mask])
        else:
            aux_mat = gs.matmul(gs.transpose(mat, axes=(0, 2, 1)), mat)

            inv_sqrt_mat = gs.zeros_like(mat)
            for i in range(n_mats):
                sym_mat = aux_mat[i]

                assert spd_matrices_space.is_symmetric(sym_mat)
                inv_sqrt_mat[i] = gs.linalg.inv(
                    spd_matrices_space.sqrtm(sym_mat))
            rot_mat = gs.matmul(mat, inv_sqrt_mat)

        assert gs.ndim(rot_mat) == 3
        return rot_mat

    def skew_matrix_from_vector(self, vec):
        """
        In 3D, compute the skew-symmetric matrix,
        known as the cross-product of a vector,
        associated to the vector vec.

        In nD, fill a skew-symmetric matrix with
        the values of the vector.
        """
        vec = gs.to_ndarray(vec, to_ndim=2)
        n_vecs, vec_dim = vec.shape

        mat_dim = int((1 + gs.sqrt(1 + 8 * vec_dim)) / 2)
        assert mat_dim == self.n

        skew_mat = gs.zeros((n_vecs,) + (self.n,) * 2)
        if self.n == 3:
            for i in range(n_vecs):
                skew_mat[i] = gs.cross(gs.eye(self.n), vec[i])
        else:
            upper_triangle_indices = gs.triu_indices(mat_dim, k=1)
            for i in range(n_vecs):
                skew_mat[i][upper_triangle_indices] = vec[i]
                skew_mat[i] = skew_mat[i] - skew_mat[i].transpose()
        assert gs.ndim(skew_mat) == 3
        return skew_mat

    def vector_from_skew_matrix(self, skew_mat):
        """
        In 3D, compute the vector defining the cross product
        associated to the skew-symmetric matrix skew mat.

        In nD, fill a vector by reading the values
        of the upper triangle of skew_mat.
        """
        skew_mat = gs.to_ndarray(skew_mat, to_ndim=3)
        n_skew_mats, mat_dim_1, mat_dim_2 = skew_mat.shape

        assert mat_dim_1 == mat_dim_2 == self.n

        vec_dim = self.dimension
        vec = gs.zeros((n_skew_mats, vec_dim))

        if self.n == 3:
            vec[:] = skew_mat[:, (2, 0, 1), (1, 2, 0)]
        else:
            idx = 0
            for j in range(mat_dim_1):
                for i in range(j):
                    vec[:, idx] = skew_mat[:, i, j]
                    idx += 1

        assert gs.ndim(vec) == 2
        return vec

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
        """
        rot_mat = gs.to_ndarray(rot_mat, to_ndim=3)
        n_rot_mats, mat_dim_1, mat_dim_2 = rot_mat.shape
        assert mat_dim_1 == mat_dim_2 == self.n

        rot_mat = self.projection(rot_mat)

        if self.n == 3:
            trace = gs.trace(rot_mat, axis1=1, axis2=2)
            trace = gs.to_ndarray(trace, to_ndim=2, axis=1)
            assert trace.shape == (n_rot_mats, 1), trace.shape

            cos_angle = .5 * (trace - 1)
            cos_angle = gs.clip(cos_angle, -1, 1)
            angle = gs.arccos(cos_angle)

            rot_mat_transpose = gs.transpose(rot_mat, axes=(0, 2, 1))
            rot_vec = self.vector_from_skew_matrix(rot_mat - rot_mat_transpose)

            mask_0 = gs.isclose(angle, 0)
            mask_0 = gs.squeeze(mask_0, axis=1)
            rot_vec[mask_0] = (rot_vec[mask_0]
                               * (.5 - (trace[mask_0] - 3.) / 12.))

            mask_pi = gs.isclose(angle, gs.pi)
            mask_pi = gs.squeeze(mask_pi, axis=1)

            # choose the largest diagonal element
            # to avoid a square root of a negative number
            a = gs.array(0)
            if gs.any(mask_pi):
                a = gs.argmax(gs.diagonal(rot_mat[mask_pi], axis1=1, axis2=2))
            b = (a + 1) % 3
            c = (a + 2) % 3

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
            rot_vec = self.vector_from_skew_matrix(skew_mat)

        return self.regularize(rot_vec, point_type='vector')

    def matrix_from_rotation_vector(self, rot_vec):
        """
        Convert rotation vector to rotation matrix.
        """
        assert self.belongs(rot_vec, point_type='vector')
        rot_vec = self.regularize(rot_vec, point_type='vector')
        n_rot_vecs, _ = rot_vec.shape

        if self.n == 3:
            angle = gs.linalg.norm(rot_vec, axis=1)
            angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

            skew_rot_vec = self.skew_matrix_from_vector(rot_vec)

            coef_1 = gs.zeros_like(angle)
            coef_2 = gs.zeros_like(angle)

            mask_0 = gs.isclose(angle, 0.0)
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

            rot_mat = self.projection(rot_mat)

        else:
            skew_mat = self.skew_matrix_from_vector(rot_vec)
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

        assert gs.ndim(quaternion) == 2
        return quaternion

    def quaternion_from_rotation_vector(self, rot_vec):
        """
        Convert a rotation vector into a unit quaternion.
        """
        assert self.n == 3, ('The quaternion representation does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        rot_vec = self.regularize(rot_vec, point_type='vector')
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

        rot_vec = self.regularize(rot_vec, point_type='vector')
        return rot_vec

    def matrix_from_quaternion(self, quaternion):
        """
        Convert a unit quaternion into a rotation vector.
        """
        assert self.n == 3, ('The quaternion representation does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        quaternion = gs.to_ndarray(quaternion, to_ndim=2)
        n_quaternions, _ = quaternion.shape

        w, x, y, z = gs.hsplit(quaternion, 4)

        rot_mat = gs.zeros((n_quaternions,) + (self.n,) * 2)

        for i in range(n_quaternions):
            # TODO(nina): vectorize by applying the composition of
            # quaternions to the identity matrix
            column_1 = [w[i] ** 2 + x[i] ** 2 - y[i] ** 2 - z[i] ** 2,
                        2 * x[i] * y[i] - 2 * w[i] * z[i],
                        2 * x[i] * z[i] + 2 * w[i] * y[i]]

            column_2 = [2 * x[i] * y[i] + 2 * w[i] * z[i],
                        w[i] ** 2 - x[i] ** 2 + y[i] ** 2 - z[i] ** 2,
                        2 * y[i] * z[i] - 2 * w[i] * x[i]]

            column_3 = [2 * x[i] * z[i] - 2 * w[i] * y[i],
                        2 * y[i] * z[i] + 2 * w[i] * x[i],
                        w[i] ** 2 - x[i] ** 2 - y[i] ** 2 + z[i] ** 2]

            rot_mat[i] = gs.hstack([column_1, column_2, column_3]).transpose()

        assert gs.ndim(rot_mat) == 3
        return rot_mat

    def matrix_from_tait_bryan_angles_extrinsic_xyz(self, tait_bryan_angles):
        """
        Convert a rotation given in terms of the tait bryan angles,
        [angle_1, angle_2, angle_3] in extrinsic (fixed) coordinate system
        in order xyz, into a rotation matrix.

        rot_mat = Z(angle_1).Y(angle_2).X(angle_3)
        where:
        - Z(angle_1) is a rotation of angle angle_1 around axis z.
        - Y(angle_2) is a rotation of angle angle_2 around axis y.
        - X(angle_3) is a rotation of angle angle_3 around axis x.
        """

        assert self.n == 3, ('The Tait-Bryan angles representation'
                             ' does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        tait_bryan_angles = gs.to_ndarray(tait_bryan_angles, to_ndim=2)
        n_tait_bryan_angles, _ = tait_bryan_angles.shape

        rot_mat = gs.zeros((n_tait_bryan_angles,) + (self.n,) * 2)
        angle_1 = tait_bryan_angles[:, 0]
        angle_2 = tait_bryan_angles[:, 1]
        angle_3 = tait_bryan_angles[:, 2]

        for i in range(n_tait_bryan_angles):
            cos_angle_1 = gs.cos(angle_1[i])
            sin_angle_1 = gs.sin(angle_1[i])
            cos_angle_2 = gs.cos(angle_2[i])
            sin_angle_2 = gs.sin(angle_2[i])
            cos_angle_3 = gs.cos(angle_3[i])
            sin_angle_3 = gs.sin(angle_3[i])

            column_1 = [[cos_angle_1 * cos_angle_2],
                        [cos_angle_2 * sin_angle_1],
                        [- sin_angle_2]]
            column_2 = [[(cos_angle_1 * sin_angle_2 * sin_angle_3
                          - cos_angle_3 * sin_angle_1)],
                        [(cos_angle_1 * cos_angle_3
                          + sin_angle_1 * sin_angle_2 * sin_angle_3)],
                        [+ cos_angle_2 * sin_angle_3]]
            column_3 = [[(sin_angle_1 * sin_angle_3
                          + cos_angle_1 * cos_angle_3 * sin_angle_2)],
                        [(cos_angle_3 * sin_angle_1 * sin_angle_2
                          - cos_angle_1 * sin_angle_3)],
                        [cos_angle_2 * cos_angle_3]]

            rot_mat[i] = gs.hstack((column_1, column_2, column_3))
        return rot_mat

    def matrix_from_tait_bryan_angles_extrinsic_zyx(self, tait_bryan_angles):
        """
        Convert a rotation given in terms of the tait bryan angles,
        [angle_1, angle_2, angle_3] in extrinsic (fixed) coordinate system
        in order zyx, into a rotation matrix.

        rot_mat = X(angle_1).Y(angle_2).Z(angle_3)
        where:
        - X(angle_1) is a rotation of angle angle_1 around axis x.
        - Y(angle_2) is a rotation of angle angle_2 around axis y.
        - Z(angle_3) is a rotation of angle angle_3 around axis z.
        """
        assert self.n == 3, ('The Tait-Bryan angles representation'
                             ' does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        tait_bryan_angles = gs.to_ndarray(tait_bryan_angles, to_ndim=2)
        n_tait_bryan_angles, _ = tait_bryan_angles.shape

        rot_mat = gs.zeros((n_tait_bryan_angles,) + (self.n,) * 2)
        angle_1 = tait_bryan_angles[:, 0]
        angle_2 = tait_bryan_angles[:, 1]
        angle_3 = tait_bryan_angles[:, 2]

        for i in range(n_tait_bryan_angles):
            cos_angle_1 = gs.cos(angle_1[i])
            sin_angle_1 = gs.sin(angle_1[i])
            cos_angle_2 = gs.cos(angle_2[i])
            sin_angle_2 = gs.sin(angle_2[i])
            cos_angle_3 = gs.cos(angle_3[i])
            sin_angle_3 = gs.sin(angle_3[i])

            column_1 = [[cos_angle_2 * cos_angle_3],
                        [(cos_angle_1 * sin_angle_3
                          + cos_angle_3 * sin_angle_1 * sin_angle_2)],
                        [(sin_angle_1 * sin_angle_3
                          - cos_angle_1 * cos_angle_3 * sin_angle_2)]]

            column_2 = [[- cos_angle_2 * sin_angle_3],
                        [(cos_angle_1 * cos_angle_3
                          - sin_angle_1 * sin_angle_2 * sin_angle_3)],
                        [(cos_angle_3 * sin_angle_1
                          + cos_angle_1 * sin_angle_2 * sin_angle_3)]]

            column_3 = [[sin_angle_2],
                        [- cos_angle_2 * sin_angle_1],
                        [cos_angle_1 * cos_angle_2]]
            rot_mat[i] = gs.hstack((column_1, column_2, column_3))
        return rot_mat

    def matrix_from_tait_bryan_angles(self, tait_bryan_angles,
                                      extrinsic_or_intrinsic='extrinsic',
                                      order='zyx'):
        """
        Convert a rotation given in terms of the tait bryan angles,
        [angle_1, angle_2, angle_3] in extrinsic (fixed) or
        intrinsic (moving) coordinate frame into a rotation matrix.

        If the order is zyx, into the rotation matrix rot_mat:
        rot_mat = X(angle_1).Y(angle_2).Z(angle_3)
        where:
        - X(angle_1) is a rotation of angle angle_1 around axis x.
        - Y(angle_2) is a rotation of angle angle_2 around axis y.
        - Z(angle_3) is a rotation of angle angle_3 around axis z.

        Exchanging 'extrinsic' and 'intrinsic' amounts to
        exchanging the order.
        """
        assert self.n == 3, ('The Tait-Bryan angles representation'
                             ' does not exist'
                             ' for rotations in %d dimensions.' % self.n)

        assert extrinsic_or_intrinsic in ('extrinsic', 'intrinsic')
        assert order in ('xyz', 'zyx')

        tait_bryan_angles = gs.to_ndarray(tait_bryan_angles, to_ndim=2)

        extrinsic_zyx = (extrinsic_or_intrinsic == 'extrinsic'
                         and order == 'zyx')
        intrinsic_xyz = (extrinsic_or_intrinsic == 'intrinsic'
                         and order == 'xyz')

        extrinsic_xyz = (extrinsic_or_intrinsic == 'extrinsic'
                         and order == 'xyz')
        intrinsic_zyx = (extrinsic_or_intrinsic == 'intrinsic'
                         and order == 'zyx')

        if extrinsic_zyx:
            rot_mat = self.matrix_from_tait_bryan_angles_extrinsic_zyx(
                tait_bryan_angles)
        elif intrinsic_xyz:
            tait_bryan_angles_reversed = gs.flip(tait_bryan_angles, axis=1)
            rot_mat = self.matrix_from_tait_bryan_angles_extrinsic_zyx(
                tait_bryan_angles_reversed)

        elif extrinsic_xyz:
            rot_mat = self.matrix_from_tait_bryan_angles_extrinsic_xyz(
                tait_bryan_angles)
        elif intrinsic_zyx:
            tait_bryan_angles_reversed = gs.flip(tait_bryan_angles, axis=1)
            rot_mat = self.matrix_from_tait_bryan_angles_extrinsic_xyz(
                tait_bryan_angles_reversed)

        else:
            raise ValueError('extrinsic_or_intrinsic should be'
                             ' \'extrinsic\' or \'intrinsic\''
                             ' and order should be \'xyz\' or \'zyx\'.')

        return rot_mat

    def tait_bryan_angles_from_matrix(self, rot_mat,
                                      extrinsic_or_intrinsic='extrinsic',
                                      order='zyx'):
        """
        Convert a rotation matrix rot_mat into the tait bryan angles,
        [angle_1, angle_2, angle_3] in extrinsic (fixed) coordinate frame,
        for the order zyx, i.e.:
        rot_mat = X(angle_1).Y(angle_2).Z(angle_3)
        where:
        - X(angle_1) is a rotation of angle angle_1 around axis x.
        - Y(angle_2) is a rotation of angle angle_2 around axis y.
        - Z(angle_3) is a rotation of angle angle_3 around axis z.
        """
        assert extrinsic_or_intrinsic in ('extrinsic', 'intrinsic')
        assert order in ('xyz', 'zyx')

        rot_mat = gs.to_ndarray(rot_mat, to_ndim=3)
        quaternion = self.quaternion_from_matrix(rot_mat)
        tait_bryan_angles = self.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        return tait_bryan_angles

    def quaternion_from_tait_bryan_angles_intrinsic_xyz(
            self, tait_bryan_angles):
        """
        Convert a rotation given by Tait-Bryan angles in extrinsic
        coordinate systems and order xyz into a unit quaternion.
        """
        assert self.n == 3, ('The quaternion representation'
                             ' and the Tait-Bryan angles representation'
                             ' do not exist'
                             ' for rotations in %d dimensions.' % self.n)
        tait_bryan_angles = gs.to_ndarray(tait_bryan_angles, to_ndim=2)
        n_tait_bryan_angles, _ = tait_bryan_angles.shape
        quaternion = gs.zeros((n_tait_bryan_angles, 4))

        matrix = self.matrix_from_tait_bryan_angles(
                tait_bryan_angles,
                extrinsic_or_intrinsic='intrinsic',
                order='xyz')
        quaternion = self.quaternion_from_matrix(matrix)
        return quaternion

    def quaternion_from_tait_bryan_angles(self, tait_bryan_angles,
                                          extrinsic_or_intrinsic='extrinsic',
                                          order='zyx'):
        """
        Convert a rotation given by Tait-Bryan angles
        into a unit quaternion.
        """
        assert extrinsic_or_intrinsic in ('extrinsic', 'intrinsic')
        assert order in ('xyz', 'zyx')

        assert self.n == 3, ('The quaternion representation'
                             ' and the Tait-Bryan angles representation'
                             ' do not exist'
                             ' for rotations in %d dimensions.' % self.n)
        tait_bryan_angles = gs.to_ndarray(tait_bryan_angles, to_ndim=2)
        n_tait_bryan_angles, _ = tait_bryan_angles.shape

        extrinsic_zyx = (extrinsic_or_intrinsic == 'extrinsic'
                         and order == 'zyx')
        intrinsic_xyz = (extrinsic_or_intrinsic == 'intrinsic'
                         and order == 'xyz')

        extrinsic_xyz = (extrinsic_or_intrinsic == 'extrinsic'
                         and order == 'xyz')
        intrinsic_zyx = (extrinsic_or_intrinsic == 'intrinsic'
                         and order == 'zyx')

        if extrinsic_zyx:
            tait_bryan_angles_reversed = gs.flip(tait_bryan_angles, axis=1)
            quat = self.quaternion_from_tait_bryan_angles_intrinsic_xyz(
                tait_bryan_angles_reversed)

        elif intrinsic_xyz:
            quat = self.quaternion_from_tait_bryan_angles_intrinsic_xyz(
                tait_bryan_angles)

        elif extrinsic_xyz:
            # TODO(nina): Put a direct implementation here,
            # instead of converting to matrices first
            rot_mat = self.matrix_from_tait_bryan_angles_extrinsic_xyz(
                tait_bryan_angles)
            quat = self.quaternion_from_matrix(rot_mat)

        elif intrinsic_zyx:
            # TODO(nina): Put a direct implementation here,
            # instead of converting to matrices first
            tait_bryan_angles_reversed = gs.flip(tait_bryan_angles, axis=1)
            rot_mat = self.matrix_from_tait_bryan_angles_extrinsic_xyz(
                tait_bryan_angles_reversed)
            quat = self.quaternion_from_matrix(rot_mat)
        else:
            raise ValueError('extrinsic_or_intrinsic should be'
                             ' \'extrinsic\' or \'intrinsic\''
                             ' and order should be \'xyz\' or \'zyx\'.')

        return quat

    def rotation_vector_from_tait_bryan_angles(
            self,
            tait_bryan_angles,
            extrinsic_or_intrinsic='extrinsic',
            order='zyx'):
        """
        Convert a rotation given by the angle_1, angle_2, angle_3
        into a rotation vector (axis-angle representation).
        """
        assert self.n == 3, ('The Tait-Bryan angles representation'
                             ' does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        assert extrinsic_or_intrinsic in ('extrinsic', 'intrinsic')
        assert order in ('xyz', 'zyx')

        quaternion = self.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        rot_vec = self.rotation_vector_from_quaternion(quaternion)

        rot_vec = self.regularize(rot_vec, point_type='vector')
        return rot_vec

    def tait_bryan_angles_from_quaternion_intrinsic_zyx(self, quaternion):
        assert self.n == 3, ('The quaternion representation'
                             ' and the Tait-Bryan angles representation'
                             ' do not exist'
                             ' for rotations in %d dimensions.' % self.n)
        quaternion = gs.to_ndarray(quaternion, to_ndim=2)

        w, x, y, z = gs.hsplit(quaternion, 4)
        angle_1 = gs.arctan2(y * z + w * x,
                             1. / 2. - (x ** 2 + y ** 2))
        angle_2 = gs.arcsin(- 2. * (x * z - w * y))
        angle_3 = gs.arctan2(x * y + w * z,
                             1. / 2. - (y ** 2 + z ** 2))
        tait_bryan_angles = gs.concatenate(
            [angle_1, angle_2, angle_3], axis=1)
        return tait_bryan_angles

    def tait_bryan_angles_from_quaternion_intrinsic_xyz(self, quaternion):
        assert self.n == 3, ('The quaternion representation'
                             ' and the Tait-Bryan angles representation'
                             ' do not exist'
                             ' for rotations in %d dimensions.' % self.n)
        quaternion = gs.to_ndarray(quaternion, to_ndim=2)

        w, x, y, z = gs.hsplit(quaternion, 4)

        angle_1 = gs.arctan2(2. * (- x * y + w * z),
                             w * w + x * x - y * y - z * z)
        angle_2 = gs.arcsin(2 * (x * z + w * y))
        angle_3 = gs.arctan2(2. * (- y * z + w * x),
                             w * w + z * z - x * x - y * y)

        tait_bryan_angles = gs.concatenate(
            [angle_1, angle_2, angle_3], axis=1)
        return tait_bryan_angles

    def tait_bryan_angles_from_quaternion(
            self, quaternion, extrinsic_or_intrinsic='extrinsic', order='zyx'):
        """
        Convert a quaternion
        to a rotation given by the angle_1, angle_2, angle_3.
        """
        assert self.n == 3, ('The quaternion representation'
                             ' and the Tait-Bryan angles representation'
                             ' do not exist'
                             ' for rotations in %d dimensions.' % self.n)
        assert extrinsic_or_intrinsic in ('extrinsic', 'intrinsic')
        assert order in ('xyz', 'zyx')

        quaternion = gs.to_ndarray(quaternion, to_ndim=2)

        extrinsic_zyx = (extrinsic_or_intrinsic == 'extrinsic'
                         and order == 'zyx')
        intrinsic_xyz = (extrinsic_or_intrinsic == 'intrinsic'
                         and order == 'xyz')

        extrinsic_xyz = (extrinsic_or_intrinsic == 'extrinsic'
                         and order == 'xyz')
        intrinsic_zyx = (extrinsic_or_intrinsic == 'intrinsic'
                         and order == 'zyx')

        if extrinsic_zyx:
            tait_bryan = self.tait_bryan_angles_from_quaternion_intrinsic_xyz(
                quaternion)
            tait_bryan = gs.flip(tait_bryan, axis=1)
        elif intrinsic_xyz:
            tait_bryan = self.tait_bryan_angles_from_quaternion_intrinsic_xyz(
                quaternion)

        elif extrinsic_xyz:
            tait_bryan = self.tait_bryan_angles_from_quaternion_intrinsic_zyx(
                quaternion)
            tait_bryan = gs.flip(tait_bryan, axis=1)
        elif intrinsic_zyx:
            tait_bryan = self.tait_bryan_angles_from_quaternion_intrinsic_zyx(
                quaternion)

        else:
            raise ValueError('extrinsic_or_intrinsic should be'
                             ' \'extrinsic\' or \'intrinsic\''
                             ' and order should be \'xyz\' or \'zyx\'.')

        return tait_bryan

    def tait_bryan_angles_from_rotation_vector(
            self, rot_vec, extrinsic_or_intrinsic='extrinsic', order='zyx'):
        """
        Convert a rotation vector (axis-angle representation)
        to a rotation given by the Tait-Bryan angles.
        """
        assert self.n == 3, ('The Tait-Bryan angles representation'
                             ' does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        assert extrinsic_or_intrinsic in ('extrinsic', 'intrinsic')
        assert order in ('xyz', 'zyx')

        rot_vec = gs.to_ndarray(rot_vec, to_ndim=2)

        quaternion = self.quaternion_from_rotation_vector(rot_vec)
        tait_bryan_angles = self.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        return tait_bryan_angles

    def compose(self, point_1, point_2, point_type=None):
        """
        Compose two elements of SO(n).
        """
        if point_type is None:
            point_type = self.default_point_type

        point_1 = self.regularize(point_1, point_type=point_type)
        point_2 = self.regularize(point_2, point_type=point_type)

        if point_type == 'vector':
            point_1 = self.matrix_from_rotation_vector(point_1)
            point_2 = self.matrix_from_rotation_vector(point_2)

        point_prod = gs.einsum('ijk,ikl->ijl', point_1, point_2)

        if point_type == 'vector':
            point_prod = self.rotation_vector_from_matrix(point_prod)

        point_prod = self.regularize(
            point_prod, point_type=point_type)
        return point_prod

    def inverse(self, point, point_type=None):
        """
        Compute the group inverse in SO(n).
        """

        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            if self.n == 3:
                inv_point = -self.regularize(point, point_type=point_type)
                return inv_point
            else:
                point = self.matrix_from_rotation_vector(point)

        inv_point = gs.linalg.inv(point)

        if point_type == 'vector':
            inv_point = self.rotation_vector_from_matrix(inv_point)

        return inv_point

    def jacobian_translation(
            self, point, left_or_right='left', point_type=None):
        """
        Compute the jacobian matrix of the differential
        of the left/right translations from the identity to point in SO(n).
        """
        assert left_or_right in ('left', 'right')

        if point_type is None:
            point_type = self.default_point_type
        assert self.belongs(point, point_type)

        if point_type == 'vector':
            if self.n == 3:
                point = self.regularize(
                    point, point_type=point_type)

                n_points, _ = point.shape

                angle = gs.linalg.norm(point, axis=1)
                angle = gs.expand_dims(angle, axis=1)

                coef_1 = gs.zeros([n_points, 1])
                coef_2 = gs.zeros([n_points, 1])

                mask_0 = gs.isclose(angle, 0)
                mask_0 = gs.squeeze(mask_0, axis=1)
                coef_1[mask_0] = (
                        TAYLOR_COEFFS_1_AT_0[0]
                        + TAYLOR_COEFFS_1_AT_0[2] * angle[mask_0] ** 2
                        + TAYLOR_COEFFS_1_AT_0[4] * angle[mask_0] ** 4
                        + TAYLOR_COEFFS_1_AT_0[6] * angle[mask_0] ** 6)
                coef_2[mask_0] = (
                        TAYLOR_COEFFS_2_AT_0[0]
                        + TAYLOR_COEFFS_2_AT_0[2] * angle[mask_0] ** 2
                        + TAYLOR_COEFFS_2_AT_0[4] * angle[mask_0] ** 4
                        + TAYLOR_COEFFS_2_AT_0[6] * angle[mask_0] ** 6)

                mask_pi = gs.isclose(angle, gs.pi)
                mask_pi = gs.squeeze(mask_pi, axis=1)
                delta_angle = angle[mask_pi] - gs.pi
                coef_1[mask_pi] = (
                        TAYLOR_COEFFS_1_AT_PI[1] * delta_angle
                        + TAYLOR_COEFFS_1_AT_PI[2] * delta_angle ** 2
                        + TAYLOR_COEFFS_1_AT_PI[3] * delta_angle ** 3
                        + TAYLOR_COEFFS_1_AT_PI[4] * delta_angle ** 4
                        + TAYLOR_COEFFS_1_AT_PI[5] * delta_angle ** 5
                        + TAYLOR_COEFFS_1_AT_PI[6] * delta_angle ** 6)

                coef_2[mask_pi] = (1 - coef_1[mask_pi]) / angle[mask_pi] ** 2

                mask_else = ~mask_0 & ~mask_pi
                coef_1[mask_else] = ((angle[mask_else] / 2)
                                     / gs.tan(angle[mask_else] / 2))
                coef_2[mask_else] = ((1 - coef_1[mask_else])
                                     / angle[mask_else] ** 2)

                jacobian = gs.zeros((n_points, self.dimension, self.dimension))
                for i in range(n_points):
                    sign = - 1
                    if left_or_right == 'left':
                        sign = + 1

                    jacobian[i] = (
                        coef_1[i] * gs.identity(self.dimension)
                        + coef_2[i] * gs.outer(point[i], point[i])
                        + sign * self.skew_matrix_from_vector(point[i]) / 2)

            else:
                if left_or_right == 'right':
                    raise NotImplementedError(
                        'The jacobian of the right translation'
                        ' is not implemented.')
                jacobian = self.matrix_from_rotation_vector(point)

            assert gs.ndim(jacobian) == 3

        elif point_type == 'matrix':
            raise NotImplementedError()

        return jacobian

    def random_uniform(self, n_samples=1, point_type=None):
        """
        Sample in SO(n) with the uniform distribution.
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            random_point = gs.random.rand(n_samples, self.dimension) * 2 - 1
            random_point = self.regularize(
                random_point, point_type=point_type)
        elif point_type == 'matrix':
            # TODO(nina): does this give the uniform distribution on rotations?
            random_matrix = gs.random.rand(n_samples, self.n, self.n)
            random_point = self.projection(random_matrix)

        return random_point

    def group_exp_from_identity(self, tangent_vec, point_type=None):
        """
        Compute the group exponential of the tangent vector at the identity.
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            point = gs.to_ndarray(tangent_vec, to_ndim=2)
        elif point_type == 'matrix':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
            tangent_vec = self.vector_from_skew_matrix(tangent_vec)
            point = self.matrix_from_rotation_vector(tangent_vec)

        return point

    def group_log_from_identity(self, point, point_type=None):
        """
        Compute the group logarithm of the point at the identity.
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            tangent_vec = self.regularize(
                point, point_type=point_type)
        elif point_type == 'matrix':
            point = self.rotation_vector_from_matrix(point)
            tangent_vec = self.skew_matrix_from_vector(point)
        return tangent_vec

    def group_exponential_barycenter(
            self, points, weights=None, point_type=None):
        """
        Compute the group exponential barycenter in SO(n), which is the
        Frechet mean of the canonical bi-invariant metric on SO(n).
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            n_points = points.shape[0]
            assert n_points > 0

            if weights is None:
                weights = gs.ones((n_points, 1))

            n_weights = weights.shape[0]
            assert n_points == n_weights

            exp_bar = self.bi_invariant_metric.mean(points, weights)

            exp_bar = gs.to_ndarray(exp_bar, to_ndim=2)
            assert gs.ndim(exp_bar) == 2, gs.ndim(exp_bar)

        elif point_type == 'matrix':
            points = self.rotation_vector_from_matrix(points)
            exp_bar = self.group_exponential_barycenter(
                points, weights, point_type='vector')
            exp_bar = self.matrix_from_rotation_vector(exp_bar)

        return exp_bar
