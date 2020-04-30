"""Exposes the `SpecialOrthogonal` group class."""

import geomstats.backend as gs
import geomstats.errors
import geomstats.vectorization
from geomstats import algebra_utils
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.invariant_metric import BiInvariantMetric
from geomstats.geometry.lie_group import LieGroup
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices


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


class _SpecialOrthogonalMatrices(GeneralLinear, LieGroup):
    """Class for special orthogonal groups in matrix representation.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        super(_SpecialOrthogonalMatrices, self).__init__(
            dim=int((n * (n - 1)) / 2), default_point_type='matrix', n=n)
        self.lie_algebra = SkewSymmetricMatrices(n=n)
        self.bi_invariant_metric = BiInvariantMetric(group=self)

    def belongs(self, point):
        """Check whether point is an orthogonal matrix."""
        return self.equal(
            self.mul(point, self.transpose(point)), self.identity)

    @classmethod
    def inverse(cls, point):
        """Return the transpose matrix of point."""
        return cls.transpose(point)

    def _is_in_lie_algebra(self, tangent_vec, atol=ATOL):
        return self.lie_algebra.belongs(tangent_vec, atol=atol)

    @classmethod
    def _to_lie_algebra(cls, tangent_vec):
        """Project vector onto skew-symmetric matrices."""
        return cls.to_skew_symmetric(tangent_vec)

    def random_uniform(self, n_samples=1, tol=1e-6):
        """Sample in SO(n) from the uniform distribution.

        Parameters
        ----------
        n_samples : int, optional (1)
            Number of samples.
        tol : unused

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled on the SO(n).
        """
        if n_samples == 1:
            random_mat = gs.random.rand(self.n, self.n)
        else:
            random_mat = gs.random.rand(n_samples, self.n, self.n)
        skew = self.to_tangent(random_mat)
        return self.exp(skew)


class _SpecialOrthogonal3Vectors(LieGroup):
    """Class for the special orthogonal group SO(3) in vector representation.

    i.e. the Lie group of rotations. This class is specific to the vector
    representation of rotations. For the matrix representation use the
    SpecialOrthogonal class and set `n=3`.

    Parameters
    ----------
    epsilon : float, optional
        precision to use for calculations involving potential divison by 0 in
        rotations
        default: 0
    """

    def __init__(self, epsilon=0.):
        LieGroup.__init__(
            self, dim=3, default_point_type='vector')

        self.n = 3
        self.epsilon = epsilon
        self.bi_invariant_metric = BiInvariantMetric(group=self)

    def get_identity(self, point_type='vector'):
        """Get the identity of the group.

        Parameters
        ----------
        point_type : str,
            Point_type of the returned value. Unused here.

        Returns
        -------
        identity : array-like, shape=[3,]
        """
        identity = gs.zeros(self.dim)
        if point_type == 'matrix':
            identity = gs.eye(self.n)
        return identity

    identity = property(get_identity)

    def belongs(self, point):
        """Evaluate if a point belongs to SO(3).

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point to check whether it belongs to SO(3).

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean indicating whether point belongs to SO(3).
        """
        vec_dim = point.shape[-1]
        belongs = vec_dim == self.dim
        if point.ndim == 2:
            belongs = gs.tile([belongs], (point.shape[0],))
        return belongs

    def regularize(self, point):
        """Regularize a point to be in accordance with convention.

        In 3D, regularize the norm of the rotation vector,
        to be between 0 and pi, following the axis-angle
        representation's convention.

        If the angle angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        Parameters
        ----------
        point : array-like, shape=[...,3]

        Returns
        -------
        regularized_point : array-like, shape=[..., 3]
        """
        regularized_point = point
        angle = gs.linalg.norm(regularized_point, axis=-1)

        mask_0 = gs.isclose(angle, 0.)
        mask_not_0 = ~mask_0
        mask_pi = gs.isclose(angle, gs.pi)

        # This avoids division by 0.
        mask_0_float = gs.cast(mask_0, gs.float32) + self.epsilon
        mask_not_0_float = (
            gs.cast(mask_not_0, gs.float32)
            + self.epsilon)
        mask_pi_float = gs.cast(mask_pi, gs.float32) + self.epsilon

        k = gs.floor(angle / (2 * gs.pi) + .5)
        angle += mask_0_float

        norms_ratio = gs.zeros_like(angle)
        norms_ratio += mask_not_0_float * (
            1. - 2. * gs.pi * k / angle)
        norms_ratio += mask_0_float
        norms_ratio += mask_pi_float * (
            gs.pi / angle
            - (1. - 2. * gs.pi * k / angle))

        regularized_point = gs.einsum(
            '...,...i->...i', norms_ratio, regularized_point)

        return regularized_point

    @geomstats.vectorization.decorator(
        ['else', 'vector', 'else', 'output_point'])
    def regularize_tangent_vec_at_identity(
            self, tangent_vec, metric=None):
        """Regularize a tangent vector at the identify.

        In 3D, regularize a tangent_vector by getting its norm at the identity,
        determined by the metric, to be less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 3]]
        metric : RiemannianMetric, optional
            default: self.left_canonical_metric

        Returns
        -------
        regularized_vec : array-like, shape=[..., 3]]
        """
        if metric is None:
            metric = self.left_canonical_metric
        tangent_vec_metric_norm = metric.norm(tangent_vec)
        tangent_vec_canonical_norm = gs.linalg.norm(tangent_vec, axis=-1)

        mask_norm_0 = gs.isclose(tangent_vec_metric_norm, 0.)
        mask_canonical_norm_0 = gs.isclose(tangent_vec_canonical_norm, 0.)

        mask_0 = mask_norm_0 | mask_canonical_norm_0
        mask_else = ~mask_0

        # This avoids division by 0.
        mask_0_float = gs.cast(mask_0, gs.float32) + self.epsilon
        mask_else_float = gs.cast(mask_else, gs.float32) + self.epsilon

        regularized_vec = gs.zeros_like(tangent_vec)
        regularized_vec += gs.einsum(
            '...,...i->...i', mask_0_float, tangent_vec)

        tangent_vec_canonical_norm += mask_0_float

        coef = gs.zeros_like(tangent_vec_metric_norm)
        coef += mask_else_float * (
            tangent_vec_metric_norm
            / tangent_vec_canonical_norm)

        coef_tangent_vec = gs.einsum(
            '...,...i->...i', coef, tangent_vec)
        regularized_vec += gs.einsum(
            '...,...i->...i',
            mask_else_float,
            self.regularize(coef_tangent_vec))

        coef += mask_0_float
        regularized_vec = gs.einsum(
            '...,...i->...i', 1. / coef, regularized_vec)
        regularized_vec = gs.einsum(
            '...,...i->...i', mask_else_float, regularized_vec)
        return regularized_vec

    @geomstats.vectorization.decorator(
        ['else', 'vector', 'vector', 'else', 'output_point'])
    def regularize_tangent_vec(
            self, tangent_vec, base_point, metric=None):
        """Regularize tangent vector at a base point.

        In 3D, regularize a tangent_vector by getting the norm of its parallel
        transport to the identity, determined by the metric, less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[...,3]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 3]
            Point on the manifold.
        metric : RiemannianMetric, optional
            default: self.left_canonical_metric

        Returns
        -------
        regularized_tangent_vec : array-like,
            shape=[..., 3]
        """
        if metric is None:
            metric = self.left_canonical_metric
        base_point = self.regularize(base_point)
        n_vecs = tangent_vec.shape[0]

        jacobian = self.jacobian_translation(
            point=base_point, left_or_right=metric.left_or_right)
        jacobian = gs.array([jacobian[0]] * n_vecs)
        inv_jacobian = gs.linalg.inv(jacobian)
        inv_jacobian = gs.to_ndarray(inv_jacobian, to_ndim=3)
        tangent_vec_at_id = gs.einsum(
            '...i,...ij->...j',
            tangent_vec,
            Matrices.transpose(inv_jacobian))

        tangent_vec_at_id = self.regularize_tangent_vec_at_identity(
            tangent_vec_at_id, metric)

        jacobian = gs.to_ndarray(jacobian, to_ndim=3)
        regularized_tangent_vec = gs.einsum(
            '...i,...ij->...j',
            tangent_vec_at_id,
            Matrices.transpose(jacobian))

        return regularized_tangent_vec

    @geomstats.vectorization.decorator(['else', 'matrix'])
    def projection(self, point):
        """Project a matrix on SO(n) using the Frobenius norm.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]

        Returns
        -------
        rot_mat : array-like, shape=[..., n, n]
        """
        mat = point
        n_mats, n, _ = mat.shape

        if n == 3:
            mat_unitary_u, _, mat_unitary_v = gs.linalg.svd(mat)
            rot_mat = gs.einsum('nij,njk->nik', mat_unitary_u, mat_unitary_v)
            mask = gs.less(gs.linalg.det(rot_mat), 0.)
            mask_float = gs.cast(mask, gs.float32) + self.epsilon
            diag = gs.array([[1., 1., -1.]])
            diag = gs.to_ndarray(
                algebra_utils.from_vector_to_diagonal_matrix(diag),
                to_ndim=3) + self.epsilon
            new_mat_diag_s = gs.tile(diag, [n_mats, 1, 1])

            aux_mat = gs.einsum(
                'nij,njk->nik',
                mat_unitary_u,
                new_mat_diag_s)
            rot_mat += gs.einsum(
                'n,njk->njk',
                mask_float,
                gs.einsum(
                    'nij,njk->nik',
                    aux_mat,
                    mat_unitary_v))
        else:
            aux_mat = gs.matmul(gs.transpose(mat, axes=(0, 2, 1)), mat)

            inv_sqrt_mat = gs.linalg.inv(
                gs.linalg.sqrtm(aux_mat))

            rot_mat = gs.matmul(mat, inv_sqrt_mat)

        return rot_mat

    @geomstats.vectorization.decorator(['else', 'vector'])
    def skew_matrix_from_vector(self, vec):
        """Get the skew-symmetric matrix derived from the vector.

        In 3D, compute the skew-symmetric matrix,known as the cross-product of
        a vector, associated to the vector `vec`.

        In nD, fill a skew-symmetric matrix with the values of the vector.

        Parameters
        ----------
        vec : array-like, shape=[..., dim]

        Returns
        -------
        skew_mat : array-like, shape=[..., n, n]
        """
        n_vecs, vec_dim = gs.shape(vec)

        if self.n == 2:
            vec = gs.tile(vec, [1, 2])
            vec = gs.reshape(vec, (n_vecs, 2))

            id_skew = gs.array(
                gs.tile([[[0., 1.], [-1., 0.]]], (n_vecs, 1, 1)))
            skew_mat = gs.einsum(
                '...ij,...i->...ij', gs.cast(id_skew, gs.float32), vec)

        elif self.n == 3:
            levi_civita_symbol = gs.tile([[
                [[0., 0., 0.],
                 [0., 0., 1.],
                 [0., -1., 0.]],
                [[0., 0., -1.],
                 [0., 0., 0.],
                 [1., 0., 0.]],
                [[0., 1., 0.],
                 [-1., 0., 0.],
                 [0., 0., 0.]]
            ]], (n_vecs, 1, 1, 1))

            levi_civita_symbol = gs.array(levi_civita_symbol)
            levi_civita_symbol += self.epsilon

            # This avoids dividing by 0.
            basis_vec_1 = gs.array(
                gs.tile([[1., 0., 0.]], (n_vecs, 1))) + self.epsilon
            basis_vec_2 = gs.array(
                gs.tile([[0., 1., 0.]], (n_vecs, 1))) + self.epsilon
            basis_vec_3 = gs.array(
                gs.tile([[0., 0., 1.]], (n_vecs, 1))) + self.epsilon

            cross_prod_1 = gs.einsum(
                'nijk,ni,nj->nk',
                levi_civita_symbol,
                basis_vec_1,
                vec)
            cross_prod_2 = gs.einsum(
                'nijk,ni,nj->nk',
                levi_civita_symbol,
                basis_vec_2,
                vec)
            cross_prod_3 = gs.einsum(
                'nijk,ni,nj->nk',
                levi_civita_symbol,
                basis_vec_3,
                vec)

            cross_prod_1 = gs.to_ndarray(cross_prod_1, to_ndim=3, axis=1)
            cross_prod_2 = gs.to_ndarray(cross_prod_2, to_ndim=3, axis=1)
            cross_prod_3 = gs.to_ndarray(cross_prod_3, to_ndim=3, axis=1)
            skew_mat = gs.concatenate(
                [cross_prod_1, cross_prod_2, cross_prod_3], axis=1)

        else:  # SO(n)
            mat_dim = gs.cast(
                ((1. + gs.sqrt(1. + 8. * vec_dim)) / 2.), gs.int32)
            skew_mat = gs.zeros((n_vecs,) + (self.n,) * 2)
            upper_triangle_indices = gs.triu_indices(mat_dim, k=1)
            for i in range(n_vecs):
                skew_mat[i][upper_triangle_indices] = vec[i]
                skew_mat[i] = skew_mat[i] - gs.transpose(skew_mat[i])
        return skew_mat

    @geomstats.vectorization.decorator(['else', 'matrix', 'output_point'])
    def vector_from_skew_matrix(self, skew_mat):
        """Derive a vector from the skew-symmetric matrix.

        In 3D, compute the vector defining the cross product
        associated to the skew-symmetric matrix skew mat.

        Parameters
        ----------
        skew_mat : array-like, shape=[..., n, n]

        Returns
        -------
        vec : array-like, shape=[..., dim]
        """
        n_skew_mats, _, _ = skew_mat.shape

        vec_dim = self.dim
        vec = gs.zeros((n_skew_mats, vec_dim))

        if self.n == 2:  # SO(2)
            vec = skew_mat[:, 0, 1]
            vec = gs.expand_dims(vec, axis=1)

        elif self.n == 3:  # SO(3)
            vec_1 = gs.to_ndarray(skew_mat[:, 2, 1], to_ndim=2, axis=1)
            vec_2 = gs.to_ndarray(skew_mat[:, 0, 2], to_ndim=2, axis=1)
            vec_3 = gs.to_ndarray(skew_mat[:, 1, 0], to_ndim=2, axis=1)
            vec = gs.concatenate([vec_1, vec_2, vec_3], axis=1)

        return vec

    @geomstats.vectorization.decorator(['else', 'matrix', 'output_point'])
    def rotation_vector_from_matrix(self, rot_mat):
        r"""Convert rotation matrix (in 3D) to rotation vector (axis-angle).

        Get the angle through the trace of the rotation matrix:
        The eigenvalues are:
        :math:`\{1, \cos(angle) + i \sin(angle), \cos(angle) - i \sin(angle)\}`
        so that:
        :math:`trace = 1 + 2 \cos(angle), \{-1 \leq trace \leq 3\}`

        Get the rotation vector through the formula:
        :math:`S_r = \frac{angle}{(2 * \sin(angle) ) (R - R^T)}`

        For the edge case where the angle is close to pi,
        the formulation is derived by using the following equality (see the
        Axis-angle representation on Wikipedia):
        :math:`outer(r, r) = \frac{1}{2} (R + I_3)`
        In nD, the rotation vector stores the :math:`n(n-1)/2` values
        of the skew-symmetric matrix representing the rotation.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., n, n]

        Returns
        -------
        regularized_rot_vec : array-like, shape=[..., 3]
        """
        n_rot_mats, _, _ = rot_mat.shape

        trace = gs.trace(rot_mat, axis1=1, axis2=2)
        trace = gs.to_ndarray(trace, to_ndim=2, axis=1)
        trace_num = gs.clip(trace, -1, 3)
        angle = gs.arccos(0.5 * (trace_num - 1))
        rot_mat_transpose = gs.transpose(rot_mat, axes=(0, 2, 1))
        rot_vec_not_pi = self.vector_from_skew_matrix(
            rot_mat - rot_mat_transpose)
        mask_0 = gs.cast(gs.isclose(angle, 0.), gs.float32)
        mask_pi = gs.cast(gs.isclose(angle, gs.pi, atol=1e-2), gs.float32)
        mask_else = (1 - mask_0) * (1 - mask_pi)

        numerator = 0.5 * mask_0 + angle * mask_else
        denominator = (1 - angle ** 2 / 6) * mask_0 + 2 * gs.sin(
            angle) * mask_else + mask_pi

        rot_vec_not_pi = rot_vec_not_pi * numerator / denominator

        vector_outer = 0.5 * (gs.eye(3) + rot_mat)
        gs.set_diag(
            vector_outer, gs.maximum(
                0., gs.diagonal(vector_outer, axis1=1, axis2=2)))
        squared_diag_comp = gs.diagonal(vector_outer, axis1=1, axis2=2)
        diag_comp = gs.sqrt(squared_diag_comp)
        norm_line = gs.linalg.norm(vector_outer, axis=2)
        max_line_index = gs.argmax(norm_line, axis=1)
        selected_line = gs.get_slice(
            vector_outer, (range(n_rot_mats), max_line_index))
        signs = gs.sign(selected_line)
        rot_vec_pi = angle * signs * diag_comp

        rot_vec = rot_vec_not_pi + mask_pi * rot_vec_pi

        return self.regularize(rot_vec)

    @geomstats.vectorization.decorator(['else', 'vector'])
    def matrix_from_rotation_vector(self, rot_vec):
        """Convert rotation vector to rotation matrix.

        Parameters
        ----------
        rot_vec: array-like, shape=[..., 3]

        Returns
        -------
        rot_mat: array-like, shape=[..., 3]
        """
        rot_vec = self.regularize(rot_vec)

        angle = gs.linalg.norm(rot_vec, axis=1)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

        skew_rot_vec = self.skew_matrix_from_vector(rot_vec)

        coef_1 = gs.zeros_like(angle)
        coef_2 = gs.zeros_like(angle)

        # This avoids dividing by 0.
        mask_0 = gs.isclose(angle, 0.)
        mask_0_float = gs.cast(mask_0, gs.float32) + self.epsilon

        coef_1 += mask_0_float * (1. - (angle ** 2) / 6.)
        coef_2 += mask_0_float * (1. / 2. - angle ** 2)

        # This avoids dividing by 0.
        mask_else = ~mask_0
        mask_else_float = gs.cast(mask_else, gs.float32) + self.epsilon

        angle += mask_0_float

        coef_1 += mask_else_float * (gs.sin(angle) / angle)
        coef_2 += mask_else_float * (
            (1. - gs.cos(angle)) / (angle ** 2))

        coef_1 = gs.squeeze(coef_1, axis=1)
        coef_2 = gs.squeeze(coef_2, axis=1)
        term_1 = (gs.eye(self.dim)
                  + gs.einsum('n,njk->njk', coef_1, skew_rot_vec))

        squared_skew_rot_vec = gs.einsum(
            'nij,njk->nik', skew_rot_vec, skew_rot_vec)

        term_2 = gs.einsum('n,njk->njk', coef_2, squared_skew_rot_vec)

        return term_1 + term_2

    @geomstats.vectorization.decorator(['else', 'matrix'])
    def quaternion_from_matrix(self, rot_mat):
        """Convert a rotation matrix into a unit quaternion.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., 3, 3]

        Returns
        -------
        quaternion : array-like, shape=[..., 4]
        """
        rot_vec = self.rotation_vector_from_matrix(rot_mat)
        quaternion = self.quaternion_from_rotation_vector(rot_vec)

        return quaternion

    @geomstats.vectorization.decorator(['else', 'vector'])
    def quaternion_from_rotation_vector(self, rot_vec):
        """Convert a rotation vector into a unit quaternion.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]

        Returns
        -------
        quaternion : array-like, shape=[..., 4]
        """
        rot_vec = self.regularize(rot_vec)

        angle = gs.linalg.norm(rot_vec, axis=1)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

        mask_0 = gs.isclose(angle, 0.)
        mask_not_0 = ~mask_0

        rotation_axis = gs.divide(
            rot_vec,
            angle
            * gs.cast(mask_not_0, gs.float32)
            + gs.cast(mask_0, gs.float32))

        quaternion = gs.concatenate(
            (gs.cos(angle / 2),
             gs.sin(angle / 2) * rotation_axis[:]),
            axis=1)

        return quaternion

    @geomstats.vectorization.decorator(['else', 'vector'])
    def rotation_vector_from_quaternion(self, quaternion):
        """Convert a unit quaternion into a rotation vector.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]

        Returns
        -------
        rot_vec : array-like, shape=[..., 3]
        """
        cos_half_angle = quaternion[:, 0]
        cos_half_angle = gs.clip(cos_half_angle, -1, 1)
        half_angle = gs.arccos(cos_half_angle)

        half_angle = gs.to_ndarray(half_angle, to_ndim=2, axis=1)

        mask_0 = gs.isclose(half_angle, 0.)
        mask_not_0 = ~mask_0

        rotation_axis = gs.divide(
            quaternion[:, 1:],
            gs.sin(half_angle) *
            gs.cast(mask_not_0, gs.float32)
            + gs.cast(mask_0, gs.float32))
        rot_vec = gs.array(
            2 * half_angle
            * rotation_axis
            * gs.cast(mask_not_0, gs.float32))

        rot_vec = self.regularize(rot_vec)
        return rot_vec

    @geomstats.vectorization.decorator(['else', 'vector'])
    def matrix_from_quaternion(self, quaternion):
        """Convert a unit quaternion into a rotation vector.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]

        Returns
        -------
        rot_mat : array-like, shape=[..., 3]
        """
        n_quaternions, _ = quaternion.shape

        w, x, y, z = gs.hsplit(quaternion, 4)

        rot_mat = gs.zeros((n_quaternions,) + (self.n,) * 2)

        for i in range(n_quaternions):
            # TODO (nina): Vectorize by applying the composition of
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

            mask_i = gs.get_mask_i_float(i, n_quaternions)
            rot_mat_i = gs.transpose(
                gs.hstack([column_1, column_2, column_3]))
            rot_mat_i = gs.to_ndarray(rot_mat_i, to_ndim=3)
            rot_mat += gs.einsum('...,...ij->...ij', mask_i, rot_mat_i)

        return rot_mat

    @geomstats.vectorization.decorator(['else', 'vector'])
    def matrix_from_tait_bryan_angles_extrinsic_xyz(self, tait_bryan_angles):
        """Convert Tait-Bryan angles to rot mat in extrensic coords (xyz).

        Convert a rotation given in terms of the tait bryan angles,
        [angle_1, angle_2, angle_3] in extrinsic (fixed) coordinate system
        in order xyz, into a rotation matrix.

        rot_mat = Z(angle_1).Y(angle_2).X(angle_3)
        where:
        - Z(angle_1) is a rotation of angle angle_1 around axis z.
        - Y(angle_2) is a rotation of angle angle_2 around axis y.
        - X(angle_3) is a rotation of angle angle_3 around axis x.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]

        Returns
        -------
        rot_mat : array-like, shape=[..., 3, 3]
        """
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
                        [cos_angle_2 * sin_angle_3]]
            column_3 = [[(sin_angle_1 * sin_angle_3
                          + cos_angle_1 * cos_angle_3 * sin_angle_2)],
                        [(cos_angle_3 * sin_angle_1 * sin_angle_2
                          - cos_angle_1 * sin_angle_3)],
                        [cos_angle_2 * cos_angle_3]]

            rot_mat[i] = gs.hstack((column_1, column_2, column_3))
        return rot_mat

    @geomstats.vectorization.decorator(['else', 'vector'])
    def matrix_from_tait_bryan_angles_extrinsic_zyx(self, tait_bryan_angles):
        """Convert Tait-Bryan angles to rot mat in extrensic coords (zyx).

        Convert a rotation given in terms of the tait bryan angles,
        [angle_1, angle_2, angle_3] in extrinsic (fixed) coordinate system
        in order zyx, into a rotation matrix.

        rot_mat = X(angle_1).Y(angle_2).Z(angle_3)
        where:
        - X(angle_1) is a rotation of angle angle_1 around axis x.
        - Y(angle_2) is a rotation of angle angle_2 around axis y.
        - Z(angle_3) is a rotation of angle angle_3 around axis z.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]

        Returns
        -------
        rot_mat : array-like, shape=[..., n, n]
        """
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

    @geomstats.vectorization.decorator(['else', 'vector', 'else', 'else'])
    def matrix_from_tait_bryan_angles(self, tait_bryan_angles,
                                      extrinsic_or_intrinsic='extrinsic',
                                      order='zyx'):
        """Convert Tait-Bryan angles to rot mat in extr or intr coords.

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

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]
        extrinsic_or_intrinsic : str, {'extrensic', 'intrinsic'} optional
            default: 'extrinsic'
        order : str, {'xyz', 'zyx'}, optional
            default: 'zyx'

        Returns
        -------
        rot_mat : array-like, shape=[..., n, n]
        """
        geomstats.errors.check_parameter_accepted_values(
            extrinsic_or_intrinsic,
            'extrinsic_or_intrinsic',
            ['extrinsic', 'intrinsic'])
        geomstats.errors.check_parameter_accepted_values(
            order,
            'order',
            ['xyz', 'zyx'])

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

    @geomstats.vectorization.decorator(['else', 'matrix', 'else', 'else'])
    def tait_bryan_angles_from_matrix(self, rot_mat,
                                      extrinsic_or_intrinsic='extrinsic',
                                      order='zyx'):
        """Convert rot_mat into Tait-Bryan angles.

        Convert a rotation matrix rot_mat into the tait bryan angles,
        [angle_1, angle_2, angle_3] in extrinsic (fixed) coordinate frame,
        for the order zyx, i.e.:
        rot_mat = X(angle_1).Y(angle_2).Z(angle_3)
        where:
        - X(angle_1) is a rotation of angle angle_1 around axis x.
        - Y(angle_2) is a rotation of angle angle_2 around axis y.
        - Z(angle_3) is a rotation of angle angle_3 around axis z.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., n, n]
        extrinsic_or_intrinsic : str, {'extrinsic', 'intrinsic'}, optional
            default: 'extrinsic'
        order : str, {'xyz', 'zyx'}, optional
            default: 'zyx'

        Returns
        -------
        tait_bryan_angles : array-like, shape=[..., 3]
        """
        quaternion = self.quaternion_from_matrix(rot_mat)
        tait_bryan_angles = self.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        return tait_bryan_angles

    @geomstats.vectorization.decorator(['else', 'vector'])
    def quaternion_from_tait_bryan_angles_intrinsic_xyz(
            self, tait_bryan_angles):
        """Convert Tait-Bryan angles to into unit quaternion.

        Convert a rotation given by Tait-Bryan angles in extrinsic
        coordinate systems and order xyz into a unit quaternion.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]

        Returns
        -------
        quaternion : array-like, shape=[..., 4]
        """
        matrix = self.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic='intrinsic',
            order='xyz')
        quaternion = self.quaternion_from_matrix(matrix)
        return quaternion

    @geomstats.vectorization.decorator(['else', 'vector', 'else', 'else'])
    def quaternion_from_tait_bryan_angles(self, tait_bryan_angles,
                                          extrinsic_or_intrinsic='extrinsic',
                                          order='zyx'):
        """Convert a rotation given by Tait-Bryan angles into unit quaternion.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]
        extrinsic_or_intrinsic : str, {'extrinsic', 'intrinsic'}, optional
            default: 'extrinsic'
        order : str, {'xyz', 'zyx'}, optional
            default: 'zyx'

        Returns
        -------
        quat : array-like, shape=[..., 4]
        """
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
            rot_mat = self.matrix_from_tait_bryan_angles_extrinsic_xyz(
                tait_bryan_angles)
            quat = self.quaternion_from_matrix(rot_mat)

        elif intrinsic_zyx:
            tait_bryan_angles_reversed = gs.flip(tait_bryan_angles, axis=1)
            rot_mat = self.matrix_from_tait_bryan_angles_extrinsic_xyz(
                tait_bryan_angles_reversed)
            quat = self.quaternion_from_matrix(rot_mat)
        else:
            raise ValueError('extrinsic_or_intrinsic should be'
                             ' \'extrinsic\' or \'intrinsic\''
                             ' and order should be \'xyz\' or \'zyx\'.')

        return quat

    @geomstats.vectorization.decorator(['else', 'vector', 'else', 'else'])
    def rotation_vector_from_tait_bryan_angles(
            self,
            tait_bryan_angles,
            extrinsic_or_intrinsic='extrinsic',
            order='zyx'):
        """Convert rotation given by angle_1, angle_2, angle_3 into rot. vec.

        Convert into axis-angle representation.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]
        extrinsic_or_intrinsic : str, {'extrinsic', 'intrinsic'}, optional
            default: 'extrinsic'
        order : str, {'xyz', 'zyx'}, optional
            default: 'zyx'

        Returns
        -------
        rot_vec : array-like, shape=[..., 3]
        """
        quaternion = self.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        rot_vec = self.rotation_vector_from_quaternion(quaternion)

        rot_vec = self.regularize(rot_vec)
        return rot_vec

    @staticmethod
    @geomstats.vectorization.decorator(['vector'])
    def tait_bryan_angles_from_quaternion_intrinsic_zyx(quaternion):
        """Convert quaternion to tait bryan representation of order zyx.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]

        Returns
        -------
        tait_bryan_angles : array-like, shape=[..., 3]
        """
        w, x, y, z = gs.hsplit(quaternion, 4)
        angle_1 = gs.arctan2(y * z + w * x,
                             1. / 2. - (x ** 2 + y ** 2))
        angle_2 = gs.arcsin(- 2. * (x * z - w * y))
        angle_3 = gs.arctan2(x * y + w * z,
                             1. / 2. - (y ** 2 + z ** 2))
        tait_bryan_angles = gs.concatenate(
            [angle_1, angle_2, angle_3], axis=1)
        return tait_bryan_angles

    @staticmethod
    @geomstats.vectorization.decorator(['vector'])
    def tait_bryan_angles_from_quaternion_intrinsic_xyz(quaternion):
        """Convert quaternion to tait bryan representation of order xyz.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]

        Returns
        -------
        tait_bryan_angles : array-like, shape=[..., 3]
        """
        w, x, y, z = gs.hsplit(quaternion, 4)

        angle_1 = gs.arctan2(2. * (- x * y + w * z),
                             w * w + x * x - y * y - z * z)
        angle_2 = gs.arcsin(2 * (x * z + w * y))
        angle_3 = gs.arctan2(2. * (- y * z + w * x),
                             w * w + z * z - x * x - y * y)

        tait_bryan_angles = gs.concatenate(
            [angle_1, angle_2, angle_3], axis=1)
        return tait_bryan_angles

    @geomstats.vectorization.decorator(['else', 'vector', 'else', 'else'])
    def tait_bryan_angles_from_quaternion(
            self, quaternion, extrinsic_or_intrinsic='extrinsic', order='zyx'):
        """Convert quaternion to a rotation in form angle_1, angle_2, angle_3.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]
        extrinsic_or_intrinsic : str, {'extrinsic', 'intrinsic'}, optional
            default: 'extrinsic'
        order : str, {'xyz', 'zyx'}, optional
            default: 'zyx'

        Returns
        -------
        tait_bryan : array-like, shape=[..., 3]
        """
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

    @geomstats.vectorization.decorator(['else', 'vector', 'else', 'else'])
    def tait_bryan_angles_from_rotation_vector(
            self, rot_vec, extrinsic_or_intrinsic='extrinsic', order='zyx'):
        """Convert a rotation vector to a rotation given by Tait-Bryan angles.

        Here the rotation vector is in the axis-angle representation.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]
        extrinsic_or_intrinsic : str, {'extrinsic', 'intrinsic'}, optional
            default: 'extrinsic'
        order : str, {'xyz', 'zyx'}, optional
            default: 'zyx'

        Returns
        -------
        tait_bryan_angles : array-like, shape=[..., 3]
        """
        quaternion = self.quaternion_from_rotation_vector(rot_vec)
        tait_bryan_angles = self.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        return tait_bryan_angles

    def compose(self, point_a, point_b):
        """Compose two elements of SO(3).

        Parameters
        ----------
        point_a : array-like, shape=[..., 3]
        point_b : array-like, shape=[..., 3]

        Returns
        -------
        point_prod : array-like, shape=[..., 3]
        """
        point_a = self.regularize(point_a)
        point_b = self.regularize(point_b)

        point_a = self.matrix_from_rotation_vector(point_a)
        point_b = self.matrix_from_rotation_vector(point_b)
        point_prod = gs.matmul(point_a, point_b)

        point_prod = self.rotation_vector_from_matrix(point_prod)
        point_prod = self.regularize(point_prod)

        return point_prod

    def inverse(self, point):
        """Compute the group inverse in SO(3).

        Parameters
        ----------
        point : array-like, shape=[..., 3]

        Returns
        -------
        inv_point : array-like, shape=[..., 3]
        """
        return -self.regularize(point)

    @geomstats.vectorization.decorator(
        ['else', 'vector', 'else'])
    def jacobian_translation(
            self, point, left_or_right='left'):
        """Compute the jacobian matrix corresponding to translation.

        Compute the jacobian matrix of the differential
        of the left/right translations from the identity to point in SO(3).

        Parameters
        ----------
        point : array-like, shape=[..., 3]
        left_or_right : str, {'left', 'right'}, optional
            default: 'left'
        point_type : str, {'vector', 'matrix'}, optional
            default: self.default_point_type

        Returns
        -------
        jacobian : array-like, shape=[..., 3, 3]
        """
        geomstats.errors.check_parameter_accepted_values(
            left_or_right, 'left_or_right', ['left', 'right'])

        point = self.regularize(point)

        n_points, _ = point.shape

        angle = gs.linalg.norm(point, axis=-1)
        angle = gs.expand_dims(angle, axis=-1)

        coef_1 = gs.zeros([n_points, 1])
        coef_2 = gs.zeros([n_points, 1])

        # This avoids dividing by 0.
        mask_0 = gs.isclose(angle, 0.)
        mask_0_float = gs.cast(mask_0, gs.float32) + self.epsilon

        coef_1 += mask_0_float * (
            TAYLOR_COEFFS_1_AT_0[0]
            + TAYLOR_COEFFS_1_AT_0[2] * angle ** 2
            + TAYLOR_COEFFS_1_AT_0[4] * angle ** 4
            + TAYLOR_COEFFS_1_AT_0[6] * angle ** 6)

        coef_2 += mask_0_float * (
            TAYLOR_COEFFS_2_AT_0[0]
            + TAYLOR_COEFFS_2_AT_0[2] * angle ** 2
            + TAYLOR_COEFFS_2_AT_0[4] * angle ** 4
            + TAYLOR_COEFFS_2_AT_0[6] * angle ** 6)

        # This avoids dividing by 0.
        mask_pi = gs.isclose(angle, gs.pi)
        mask_pi_float = gs.cast(mask_pi, gs.float32) + self.epsilon

        delta_angle = angle - gs.pi
        coef_1 += mask_pi_float * (
            TAYLOR_COEFFS_1_AT_PI[1] * delta_angle
            + TAYLOR_COEFFS_1_AT_PI[2] * delta_angle ** 2
            + TAYLOR_COEFFS_1_AT_PI[3] * delta_angle ** 3
            + TAYLOR_COEFFS_1_AT_PI[4] * delta_angle ** 4
            + TAYLOR_COEFFS_1_AT_PI[5] * delta_angle ** 5
            + TAYLOR_COEFFS_1_AT_PI[6] * delta_angle ** 6)

        angle += mask_0_float
        coef_2 += mask_pi_float * (
            (1 - coef_1) / angle ** 2)

        # This avoids dividing by 0.
        mask_else = ~mask_0 & ~mask_pi
        mask_else_float = gs.cast(mask_else, gs.float32) + self.epsilon

        # This avoids division by 0.
        angle += mask_pi_float
        coef_1 += mask_else_float * ((angle / 2) / gs.tan(angle / 2))
        coef_2 += mask_else_float * ((1 - coef_1) / angle ** 2)
        jacobian = gs.zeros((n_points, self.dim, self.dim))
        n_points_tensor = gs.array(n_points)
        for i in range(n_points):
            # This avoids dividing by 0.
            mask_i_float = (
                gs.get_mask_i_float(i, n_points_tensor)
                + self.epsilon)

            sign = - 1.
            if left_or_right == 'left':
                sign = + 1.

            jacobian_i = (
                coef_1[i] * gs.eye(self.dim)
                + coef_2[i] * gs.outer(point[i], point[i])
                + sign * self.skew_matrix_from_vector(point[i]) / 2.)

            jacobian += gs.einsum('n,ij->nij', mask_i_float, jacobian_i)

        return jacobian

    def random_uniform(self, n_samples=1):
        """Sample in SO(3) with the uniform distribution.

        Parameters
        ----------
        n_samples : int
            the amount of samples

        Returns
        -------
        point : array-like, shape=[..., 3]
        """
        random_point = gs.random.rand(n_samples, self.dim) * 2 - 1
        random_point = self.regularize(random_point)

        if n_samples == 1:
            random_point = gs.squeeze(random_point, axis=0)

        return random_point

    @geomstats.vectorization.decorator(['else', 'vector'])
    def exp_from_identity(self, tangent_vec):
        """Compute the group exponential of the tangent vector at the identity.

        As rotations are represented by their rotation vector,
        which corresponds to the element `X` in the Lie Algebra such that
        `exp(X) = R`, this methods returns its input without change.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 3]
        point_type : str, {'vector', 'matrix'}, optional
            default: self.default_point_type

        Returns
        -------
        point : array-like, shape=[..., 3]
        """
        return tangent_vec

    @geomstats.vectorization.decorator(['else', 'vector'])
    def log_from_identity(self, point):
        """Compute the group logarithm of the point at the identity.

        As rotations are represented by their rotation vector,
        which corresponds to the element `X` in the Lie Algebra such that
        `exp(X) = R`, this methods returns its input after regularization.


        Parameters
        ----------
        point : array-like, shape=[..., 3]

        Returns
        -------
        tangent_vec : array-like, shape=[..., {dimension, [n, n]}]
        """
        return self.regularize(point)

    def lie_bracket(
            self, tangent_vector_a, tangent_vector_b, base_point=None):
        """Compute the lie bracket of two tangent vectors.

        For matrix Lie groups with tangent vectors A,B at the same base point P
        this is given by (translate to identity, compute commutator, go back)
        :math:`[A,B] = A_P^{-1}B - B_P^{-1}A`

        Parameters
        ----------
        tangent_vector_a : shape=[..., n, n]
        tangent_vector_b : shape=[..., n, n]
        base_point : array-like, shape=[..., n, n]

        Returns
        -------
        bracket : array-like, shape=[..., n, n]
        """
        return gs.cross(tangent_vector_a, tangent_vector_b)

    def exp(self, tangent_vec, base_point=None):
        return LieGroup.exp(self, tangent_vec, base_point)

    def log(self, point, base_point=None):
        return LieGroup.log(self, point, base_point)


class SpecialOrthogonal(
        _SpecialOrthogonal3Vectors, _SpecialOrthogonalMatrices):
    r"""Class for the special orthogonal groups.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices : n x n.
    point_type : str, {\'vector\', \'matrix\'}
        Representation of the elements of the group.
    epsilon : float, optional
        precision to use for calculations involving potential divison by 0 in
        rotations
        default: 0
    """

    def __new__(cls, n, point_type='matrix', epsilon=0.):
        """Instantiate a special orthogonal group.

        Select the object to instantiate depending on the point_type.
        """
        if n == 3 and point_type == 'vector':
            return _SpecialOrthogonal3Vectors(epsilon)
        if n != 3 and point_type == 'vector':
            raise NotImplementedError(
                'SO(n) is only implemented in matrix representation'
                ' when n != 3.')
        return _SpecialOrthogonalMatrices(n)
