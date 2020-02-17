"""The special orthogonal group SO(n).

i.e. the Lie group of rotations in n dimensions.
"""

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.lie_group import LieGroup
from geomstats.learning.frechet_mean import FrechetMean

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


class SpecialOrthogonal(LieGroup, EmbeddedManifold):
    """Class for the special orthogonal group SO(n).

    i.e. the Lie group of rotations.
    """

    def __init__(self, n, point_type=None, epsilon=0.):
        """Initialize an instance of SO(n).

        Parameters
        ----------
        n : int
            the dimension of the euclidean space that SO(n) acts upon
        point_type : str, {'vector', 'matrix'}, optional
            if None is given, point_type is set to 'vector for dimension 3
            and matrix otherwise
        epsilon : float, optional
            precision to use for calculations involving potential divison by in
            rotations
            default: 0
        """
        assert isinstance(n, int) and n > 1

        self.n = n
        self.dimension = int((n * (n - 1)) / 2)

        self.epsilon = epsilon

        self.default_point_type = point_type
        if point_type is None:
            self.default_point_type = 'vector' if n == 3 else 'matrix'

        LieGroup.__init__(self,
                          dimension=self.dimension)
        EmbeddedManifold.__init__(self,
                                  dimension=self.dimension,
                                  embedding_manifold=GeneralLinear(n=n))
        self.bi_invariant_metric = self.left_canonical_metric

    def get_identity(self, point_type=None):
        """Get the identity of the group.

        Parameters
        ----------
        point_type : str, {'vector', 'matrix'}, optional
            the point_type of the returned value

        Returns
        -------
        identity : array-like, shape={[dimension], [n, n]}
        """
        if point_type is None:
            point_type = self.default_point_type

        identity = gs.zeros(self.dimension)
        if point_type == 'matrix':
            identity = gs.eye(self.n)
        return identity

    identity = property(get_identity)

    def belongs(self, point, point_type=None):
        """Evaluate if a point belongs to SO(n).

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dimension, [n, n]}]
            the point of which to check whether it belongs to SO(n)
        point_type : str, {'vector', 'matrix'}, optional
            default: default_point_type

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
            array of booleans indicating whether point belongs to SO(n)
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            point = gs.to_ndarray(point, to_ndim=2)
            n_points, vec_dim = point.shape
            belongs = vec_dim == self.dimension
            belongs = gs.to_ndarray(belongs, to_ndim=1)
            belongs = gs.to_ndarray(belongs, to_ndim=2, axis=1)
            belongs = gs.tile(belongs, (n_points, 1))
            return belongs

        elif point_type == 'matrix':
            point = gs.to_ndarray(point, to_ndim=3)
            point_transpose = gs.transpose(point, axes=(0, 2, 1))
            mask = gs.isclose(gs.matmul(point, point_transpose),
                              gs.eye(self.n))
            mask = gs.all(mask, axis=(1, 2))

            mask = gs.to_ndarray(mask, to_ndim=1)
            mask = gs.to_ndarray(mask, to_ndim=2, axis=1)
            return mask

    def regularize(self, point, point_type=None):
        """Regularize a point to be in accordance with convention.

        In 3D, regularize the norm of the rotation vector,
        to be between 0 and pi, following the axis-angle
        representation's convention.

        If the angle angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dimension, [n, n]}]
        point_type : str, {'vector', 'matrix'}, optional
            default: self.default_point_type

        Returns
        -------
        regularized_point : array-like, shape=[n_samples, {dimension, [n, n]}]
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            point = gs.to_ndarray(point, to_ndim=2)
            n_points, _ = point.shape

            regularized_point = point
            if self.n == 3:
                angle = gs.linalg.norm(regularized_point, axis=1)

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
                    'n,ni->ni', norms_ratio, regularized_point)

            assert gs.ndim(regularized_point) == 2

        elif point_type == 'matrix':
            point = gs.to_ndarray(point, to_ndim=3)
            regularized_point = gs.to_ndarray(point, to_ndim=3)

        return regularized_point

    def regularize_tangent_vec_at_identity(
            self, tangent_vec, metric=None, point_type=None):
        """Regularize a tangent vector at the identify.

        In 3D, regularize a tangent_vector by getting its norm at the identity,
        determined by the metric, to be less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, {dimension, [n, n]}]
        metric : RiemannianMetric, optional
            default: self.left_canonical_metric
        point_type : str, {'vector', 'matrix'}, optional
            default: self.default_point_type

        Returns
        -------
        regularized_vec : array-like, shape=[n_samples, {dimension, [n, n]}]
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

                mask_norm_0 = gs.isclose(tangent_vec_metric_norm, 0.)
                mask_canonical_norm_0 = gs.isclose(
                    tangent_vec_canonical_norm, 0.)

                mask_0 = mask_norm_0 | mask_canonical_norm_0
                mask_else = ~mask_0

                # This avoids division by 0.
                mask_0_float = gs.cast(mask_0, gs.float32) + self.epsilon
                mask_else_float = gs.cast(mask_else, gs.float32) + self.epsilon

                regularized_vec = gs.zeros_like(tangent_vec)
                regularized_vec += mask_0_float * tangent_vec

                tangent_vec_canonical_norm += mask_0_float

                coef = gs.zeros_like(tangent_vec_metric_norm)
                coef += mask_else_float * (
                    tangent_vec_metric_norm
                    / tangent_vec_canonical_norm)
                regularized_vec += mask_else_float * self.regularize(
                    coef * tangent_vec)
                coef += mask_0_float
                regularized_vec = mask_else_float * (
                    regularized_vec / coef)
            else:
                # TODO(nina): Check if/how regularization is needed in nD?
                regularized_vec = tangent_vec

        elif point_type == 'matrix':
            regularized_vec = tangent_vec

        return regularized_vec

    def regularize_tangent_vec(
            self, tangent_vec, base_point,
            metric=None, point_type=None):
        """Regularize tangent vector at a base point.

        In 3D, regularize a tangent_vector by getting the norm of its parallel
        transport to the identity, determined by the metric, less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, {dimension, [n, n]}]
        metric : RiemannianMetric, optional
            default: self.left_canonical_metric
        point_type : str, {'vector', 'matrix'}, optional
            default: self.default_point_type

        Returns
        -------
        regularized_tangent_vec : array-like,
            shape=[n_samples, {dimension, [n, n]}]
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)

            if self.n == 3:
                if metric is None:
                    metric = self.left_canonical_metric
                base_point = self.regularize(base_point, point_type)
                n_vecs = tangent_vec.shape[0]

                jacobian = self.jacobian_translation(
                    point=base_point,
                    left_or_right=metric.left_or_right,
                    point_type=point_type)
                jacobian = gs.array([jacobian[0]] * n_vecs)
                inv_jacobian = gs.linalg.inv(jacobian)
                inv_jacobian = gs.to_ndarray(inv_jacobian, to_ndim=3)
                tangent_vec_at_id = gs.einsum(
                    'ni,nij->nj',
                    tangent_vec,
                    gs.transpose(inv_jacobian, axes=(0, 2, 1)))

                tangent_vec_at_id = self.regularize_tangent_vec_at_identity(
                    tangent_vec_at_id, metric, point_type)

                jacobian = gs.to_ndarray(jacobian, to_ndim=3)
                regularized_tangent_vec = gs.einsum(
                    'ni,nij->nj',
                    tangent_vec_at_id,
                    gs.transpose(jacobian, axes=(0, 2, 1)))
            else:
                # TODO(nina): Check if/how regularization is needed in nD?
                regularized_tangent_vec = tangent_vec

        elif point_type == 'matrix':
            regularized_tangent_vec = tangent_vec

        return regularized_tangent_vec

    def projection(self, mat):
        """Project a matrix on SO(n) using the Frobenius norm.

        Parameters
        ----------
        mat : array-like, shape=[n_samples, n, n]

        Returns
        -------
        rot_mat : array-like, shape=[n_samples, n, n]
        """
        mat = gs.to_ndarray(mat, to_ndim=3)

        n_mats, mat_dim_1, mat_dim_2 = mat.shape
        assert mat_dim_1 == mat_dim_2 == self.n

        if self.n == 3:
            mat_unitary_u, diag_s, mat_unitary_v = gs.linalg.svd(mat)
            rot_mat = gs.einsum('nij,njk->nik', mat_unitary_u, mat_unitary_v)
            mask = gs.less(gs.linalg.det(rot_mat), 0.)
            mask_float = gs.cast(mask, gs.float32) + self.epsilon
            diag = gs.array([[1., 1., -1.]])
            diag = gs.to_ndarray(gs.diag(diag), to_ndim=3) + self.epsilon
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

        assert gs.ndim(rot_mat) == 3
        return rot_mat

    def skew_matrix_from_vector(self, vec):
        """Get the skew-symmetric matrix derived from the vector.

        In 3D, compute the skew-symmetric matrix,known as the cross-product of
        a vector, associated to the vector `vec`.

        In nD, fill a skew-symmetric matrix with the values of the vector.

        Parameters
        ----------
        vec : array-like, shape=[n_samples, dimension]

        Returns
        -------
        skew_mat : array-like, shape=[n_samples, n, n]
        """
        vec = gs.to_ndarray(vec, to_ndim=2)
        n_vecs = vec.shape[0]
        vec_dim = gs.shape(vec)[1]

        if self.n == 2:  # SO(2)
            id_skew = gs.array([[[0., 1.], [-1., 0.]]] * n_vecs)
            skew_mat = gs.einsum(
                'nij,ni->nij', gs.cast(id_skew, gs.float32), vec)

        elif self.n == 3:  # SO(3)
            # This avois dividing by 0.
            levi_civita_symbol = gs.array([[
                [[0., 0., 0.],
                 [0., 0., 1.],
                 [0., -1., 0.]],
                [[0., 0., -1.],
                 [0., 0., 0.],
                 [1., 0., 0.]],
                [[0., 1., 0.],
                 [-1., 0., 0.],
                 [0., 0., 0.]]
            ]] * n_vecs) + self.epsilon

            # This avois dividing by 0.
            basis_vec_1 = gs.array([[1., 0., 0.]] * n_vecs) + self.epsilon
            basis_vec_2 = gs.array([[0., 1., 0.]] * n_vecs) + self.epsilon
            basis_vec_3 = gs.array([[0., 0., 1.]] * n_vecs) + self.epsilon
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
                skew_mat[i] = skew_mat[i] - skew_mat[i].transpose()
        assert gs.ndim(skew_mat) == 3
        return skew_mat

    def vector_from_skew_matrix(self, skew_mat):
        """Derive a vector from the skew-symmetric matrix.

        In 3D, compute the vector defining the cross product
        associated to the skew-symmetric matrix skew mat.

        In nD, fill a vector by reading the values
        of the upper triangle of skew_mat.

        Parameters
        ----------
        skew_mat : array-like, shape=[n_samples, n, n]

        Returns
        -------
        vec : array-like, shape=[n_samples, dimension]
        """
        skew_mat = gs.to_ndarray(skew_mat, to_ndim=3)
        n_skew_mats, mat_dim_1, mat_dim_2 = skew_mat.shape

        assert mat_dim_1 == mat_dim_2 == self.n

        vec_dim = self.dimension
        vec = gs.zeros((n_skew_mats, vec_dim))

        if self.n == 2:  # SO(2)
            vec = gs.expand_dims(skew_mat[:, 0, 1], axis=1)

        elif self.n == 3:  # SO(3)
            vec_1 = gs.to_ndarray(skew_mat[:, 2, 1], to_ndim=2, axis=1)
            vec_2 = gs.to_ndarray(skew_mat[:, 0, 2], to_ndim=2, axis=1)
            vec_3 = gs.to_ndarray(skew_mat[:, 1, 0], to_ndim=2, axis=1)
            vec = gs.concatenate([vec_1, vec_2, vec_3], axis=1)

        else:  # SO(n)
            idx = 0
            for j in range(mat_dim_1):
                for i in range(j):
                    vec[:, idx] = skew_mat[:, i, j]
                    idx += 1

        assert gs.ndim(vec) == 2
        return vec

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
        the formulation is derived by going from rotation matrix to unit
        quaternion to axis-angle:
        :math:`r = \frac{angle*v}{|v|}`
        where :math:`(w, v)` is a unit quaternion.

        In nD, the rotation vector stores the :math:`n(n-1)/2` values
        of the skew-symmetric matrix representing the rotation.

        Parameters
        ----------
        rot_mat : array-like, shape=[n_samples, n, n]

        Returns
        -------
        regularized_rot_vec : array-like, shape=[n_samples, dimension]
        """
        rot_mat = gs.to_ndarray(rot_mat, to_ndim=3)
        n_rot_mats, mat_dim_1, mat_dim_2 = rot_mat.shape
        assert mat_dim_1 == mat_dim_2 == self.n

        if self.n == 3:
            trace = gs.trace(rot_mat, axis1=1, axis2=2)
            trace = gs.to_ndarray(trace, to_ndim=2, axis=1)
            assert trace.shape == (n_rot_mats, 1), trace.shape

            cos_angle = .5 * (trace - 1)
            cos_angle = gs.clip(cos_angle, -1, 1)
            angle = gs.arccos(cos_angle)

            rot_mat_transpose = gs.transpose(rot_mat, axes=(0, 2, 1))
            rot_vec = self.vector_from_skew_matrix(rot_mat - rot_mat_transpose)

            # This avois dividing by 0.
            mask_0 = gs.isclose(angle, 0.)
            mask_0_float = gs.cast(mask_0, gs.float32) + self.epsilon

            rot_vec *= (1. + mask_0_float * (.5 - (trace - 3.) / 12. - 1.))

            # This avois dividing by 0.
            mask_pi = gs.isclose(angle, gs.pi)
            mask_pi_float = gs.cast(mask_pi, gs.float32) + self.epsilon

            # This avois dividing by 0.
            mask_else = ~mask_0 & ~mask_pi
            mask_else_float = gs.cast(mask_else, gs.float32) + self.epsilon

            mask_pi = gs.squeeze(mask_pi, axis=1)

            # choose the largest diagonal element
            # to avoid a square root of a negative number
            rot_mat_pi = gs.einsum(
                'ni,njk->njk', mask_pi_float, rot_mat)
            a = gs.array(0)
            rot_mat_pi_00 = gs.to_ndarray(
                rot_mat_pi[:, 0, 0], to_ndim=2, axis=1)
            rot_mat_pi_11 = gs.to_ndarray(
                rot_mat_pi[:, 1, 1], to_ndim=2, axis=1)
            rot_mat_pi_22 = gs.to_ndarray(
                rot_mat_pi[:, 2, 2], to_ndim=2, axis=1)
            rot_mat_pi_diagonal = gs.hstack(
                [rot_mat_pi_00, rot_mat_pi_11, rot_mat_pi_22])
            a = gs.argmax(rot_mat_pi_diagonal, axis=1)[0]
            b = (a + 1) % 3
            c = (a + 2) % 3

            # compute the axis vector
            sq_root = gs.zeros((n_rot_mats, 1))

            aux = gs.sqrt(
                mask_pi_float * (
                    rot_mat[:, a, a]
                    - rot_mat[:, b, b]
                    - rot_mat[:, c, c]) + 1.)
            sq_root_pi = gs.einsum(
                'ni,nk->ni', mask_pi_float, aux)

            sq_root += sq_root_pi

            rot_vec_pi = gs.zeros((n_rot_mats, self.dimension))

            # This avois dividing by 0.
            mask_a_float = gs.get_mask_i_float(a, 3) + self.epsilon
            mask_b_float = gs.get_mask_i_float(b, 3) + self.epsilon
            mask_c_float = gs.get_mask_i_float(c, 3) + self.epsilon

            mask_a_float = gs.to_ndarray(mask_a_float, to_ndim=2, axis=1)
            mask_b_float = gs.to_ndarray(mask_b_float, to_ndim=2, axis=1)
            mask_c_float = gs.to_ndarray(mask_c_float, to_ndim=2, axis=1)

            mask_a_float = gs.transpose(mask_a_float)
            mask_b_float = gs.transpose(mask_b_float)
            mask_c_float = gs.transpose(mask_c_float)

            mask_a_float = gs.tile(mask_a_float, (n_rot_mats, 1))
            mask_b_float = gs.tile(mask_b_float, (n_rot_mats, 1))
            mask_c_float = gs.tile(mask_c_float, (n_rot_mats, 1))

            rot_vec_pi += mask_pi_float * mask_a_float * sq_root / 2.

            sq_root += mask_0_float
            sq_root += mask_else_float

            rot_vec_pi_b = gs.zeros_like(rot_vec_pi)
            rot_vec_pi_c = gs.zeros_like(rot_vec_pi)

            rot_vec_pi_b += gs.einsum(
                'nk,ni->nk',
                mask_b_float,
                ((rot_mat[:, b, a]
                  + rot_mat[:, a, b])
                 / (2. * sq_root)))
            rot_vec_pi += mask_pi_float * gs.einsum(
                'ni,nk->nk', mask_pi_float, rot_vec_pi_b)

            rot_vec_pi_c += gs.einsum(
                'nk,ni->nk',
                mask_c_float,
                ((rot_mat[:, c, a]
                  + rot_mat[:, a, c])
                 / (2. * sq_root)))

            rot_vec_pi += mask_pi_float * gs.einsum(
                'ni,nk->nk',
                mask_pi_float,
                rot_vec_pi_c)

            norm_rot_vec_pi = gs.linalg.norm(rot_vec_pi, axis=1)
            norm_rot_vec_pi += gs.squeeze(mask_0_float, axis=1)
            norm_rot_vec_pi += gs.squeeze(mask_else_float, axis=1)

            rot_vec += mask_pi_float * (
                gs.einsum(
                    'nk,n->nk',
                    angle * rot_vec_pi,
                    1. / norm_rot_vec_pi))

            angle += mask_0_float
            angle = gs.to_ndarray(angle, to_ndim=2, axis=1)
            fact = gs.einsum(
                'ni,ni->ni',
                mask_else_float,
                (angle / (2. * gs.sin(angle)) - 1.))

            rot_vec *= (1. + fact)
        else:
            skew_mat = self.embedding_manifold.log(rot_mat)
            rot_vec = self.vector_from_skew_matrix(skew_mat)

        return self.regularize(rot_vec, point_type='vector')

    def matrix_from_rotation_vector(self, rot_vec):
        """Convert rotation vector to rotation matrix.

        Parameters
        ----------
        rot_vec: array-like, shape=[n_samples, dimension]

        Returns
        -------
        rot_mat: array-like, shape=[n_samples, {dimension, [n, n]}]
        """
        rot_vec = self.regularize(rot_vec, point_type='vector')
        n_rot_vecs, _ = rot_vec.shape

        if self.n == 3:
            angle = gs.linalg.norm(rot_vec, axis=1)
            angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

            skew_rot_vec = self.skew_matrix_from_vector(rot_vec)

            coef_1 = gs.zeros_like(angle)
            coef_2 = gs.zeros_like(angle)

            # This avois dividing by 0.
            mask_0 = gs.isclose(angle, 0.)
            mask_0_float = gs.cast(mask_0, gs.float32) + self.epsilon

            coef_1 += mask_0_float * (1. - (angle ** 2) / 6.)
            coef_2 += mask_0_float * (1. / 2. - angle ** 2)

            # This avois dividing by 0.
            mask_else = ~mask_0
            mask_else_float = gs.cast(mask_else, gs.float32) + self.epsilon

            angle += mask_0_float

            coef_1 += mask_else_float * (gs.sin(angle) / angle)
            coef_2 += mask_else_float * (
                (1. - gs.cos(angle)) / (angle ** 2))

            coef_1 = gs.squeeze(coef_1, axis=1)
            coef_2 = gs.squeeze(coef_2, axis=1)
            term_1 = (gs.eye(self.dimension)
                      + gs.einsum('n,njk->njk', coef_1, skew_rot_vec))

            squared_skew_rot_vec = gs.einsum(
                'nij,njk->nik', skew_rot_vec, skew_rot_vec)

            term_2 = gs.einsum('n,njk->njk', coef_2, squared_skew_rot_vec)

            rot_mat = term_1 + term_2

        else:
            skew_mat = self.skew_matrix_from_vector(rot_vec)
            rot_mat = self.embedding_manifold.exp(skew_mat)

        return rot_mat

    def quaternion_from_matrix(self, rot_mat):
        """Convert a rotation matrix into a unit quaternion.

        Parameters
        ----------
        rot_mat : array-like, shape=[n_samples, n, n]

        Returns
        -------
        quaternion : array-like, shape=[n_samples, 4]
        """
        assert self.n == 3, ('The quaternion representation does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        rot_mat = gs.to_ndarray(rot_mat, to_ndim=3)

        rot_vec = self.rotation_vector_from_matrix(rot_mat)
        quaternion = self.quaternion_from_rotation_vector(rot_vec)

        assert gs.ndim(quaternion) == 2
        return quaternion

    def quaternion_from_rotation_vector(self, rot_vec):
        """Convert a rotation vector into a unit quaternion.

        Parameters
        ----------
        rot_vec : array-like, shape=[n_samples, dimension]

        Returns
        -------
        quaternion : array-like, shape=[n_samples, 4]
        """
        assert self.n == 3, ('The quaternion representation does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        rot_vec = self.regularize(rot_vec, point_type='vector')
        n_rot_vecs, _ = rot_vec.shape

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

    def rotation_vector_from_quaternion(self, quaternion):
        """Convert a unit quaternion into a rotation vector.

        Parameters
        ----------
        quaternion : array-like, shape=[n_samples, 4]

        Returns
        -------
        rot_vec : array-like, shape=[n_samples, dimension]
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

        rot_vec = self.regularize(rot_vec, point_type='vector')
        return rot_vec

    def matrix_from_quaternion(self, quaternion):
        """Convert a unit quaternion into a rotation vector.

        Parameters
        ----------
        quaternion : array-like, shape=[n_samples, 4]

        Returns
        -------
        rot_mat : array-like, shape=[n_samples, dimension]
        """
        assert self.n == 3, ('The quaternion representation does not exist'
                             ' for rotations in %d dimensions.' % self.n)
        quaternion = gs.to_ndarray(quaternion, to_ndim=2)
        n_quaternions, _ = quaternion.shape

        w, x, y, z = gs.hsplit(quaternion, 4)

        rot_mat = gs.zeros((n_quaternions,) + (self.n,) * 2)

        for i in range(n_quaternions):
            # TODO(nina): Vectorize by applying the composition of
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
            rot_mat += gs.einsum('n,nij->nij', mask_i, rot_mat_i)

        assert gs.ndim(rot_mat) == 3
        return rot_mat

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
        tait_bryan_angles : array-like, shape=[n_samples, 3]

        Returns
        -------
        rot_mat : array-like, shape=[n_samples, n, n]
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
                        [cos_angle_2 * sin_angle_3]]
            column_3 = [[(sin_angle_1 * sin_angle_3
                          + cos_angle_1 * cos_angle_3 * sin_angle_2)],
                        [(cos_angle_3 * sin_angle_1 * sin_angle_2
                          - cos_angle_1 * sin_angle_3)],
                        [cos_angle_2 * cos_angle_3]]

            rot_mat[i] = gs.hstack((column_1, column_2, column_3))
        return rot_mat

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
        tait_bryan_angles : array-like, shape=[n_samples, 3]

        Returns
        -------
        rot_mat : array-like, shape=[n_samples, n, n]
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
        tait_bryan_angles : array-like, shape=[n_samples, 3]
        extrinsic_or_intrinsic : str, {'extrensic', 'intrinsic'} optional
            default: 'extrinsic'
        order : str, {'xyz', 'zyx'}, optional
            default: 'zyx'

        Returns
        -------
        rot_mat : array-like, shape=[n_samples, n, n]
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
        rot_mat : array-like, shape=[n_samples, n, n]
        extrinsic_or_intrinsic : str, {'extrinsic', 'intrinsic'}, optional
            default: 'extrinsic'
        order : str, {'xyz', 'zyx'}, optional
            default: 'zyx'

        Returns
        -------
        tait_bryan_angles : array-like, shape=[n_samples, 3]
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
        """Convert Tait-Bryan angles to into unit quaternion.

        Convert a rotation given by Tait-Bryan angles in extrinsic
        coordinate systems and order xyz into a unit quaternion.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[n_samples, 3]

        Returns
        -------
        quaternion : array-like, shape=[n_samples, 4]
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
        """Convert a rotation given by Tait-Bryan angles into unit quaternion.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[n_samples, 3]
        extrinsic_or_intrinsic : str, {'extrinsic', 'intrinsic'}, optional
            default: 'extrinsic'
        order : str, {'xyz', 'zyx'}, optional
            default: 'zyx'

        Returns
        -------
        quat : array-like, shape=[n_samples, 4]
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

    def rotation_vector_from_tait_bryan_angles(
            self,
            tait_bryan_angles,
            extrinsic_or_intrinsic='extrinsic',
            order='zyx'):
        """Convert rotation given by angle_1, angle_2, angle_3 into rot. vec.

        Convert into axis-angle representation.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[n_samples, 3]
        extrinsic_or_intrinsic : str, {'extrinsic', 'intrinsic'}, optional
            default: 'extrinsic'
        order : str, {'xyz', 'zyx'}, optional
            default: 'zyx'

        Returns
        -------
        rot_vec : array-like, shape=[n_samples, dimension]
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
        """Convert quaternion to tait bryan representation of order zyx.

        Parameters
        ----------
        quaternion : array-like, shape=[n_samples, 4]

        Returns
        -------
        tait_bryan_angles : array-like, shape=[n_samples, 3]
        """
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
        """Convert quaternion to tait bryan representation of order xyz.

        Parameters
        ----------
        quaternion : array-like, shape=[n_samples, 4]

        Returns
        -------
        tait_bryan_angles : array-like, shape=[n_samples, 3]
        """
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
        """Convert quaternion to a rotation in form angle_1, angle_2, angle_3.

        Parameters
        ----------
        quaternion : array-like, shape=[n_samples, 4]
        extrinsic_or_intrinsic : str, {'extrinsic', 'intrinsic'}, optional
            default: 'extrinsic'
        order : str, {'xyz', 'zyx'}, optional
            default: 'zyx'

        Returns
        -------
        tait_bryan : array-like, shape=[n_samples, 3]
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
        """Convert a rotation vector to a rotation given by Tait-Bryan angles.

        Here the rotation vector is in the axis-angle representation.

        Parameters
        ----------
        rot_vec : array-like, shape=[n_samples, dimension]
        extrinsic_or_intrinsic : str, {'extrinsic', 'intrinsic'}, optional
            default: 'extrinsic'
        order : str, {'xyz', 'zyx'}, optional
            default: 'zyx'

        Returns
        -------
        tait_bryan_angles : array-like, shape=[n_samples, 3]
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
        """Compose two elements of SO(n).

        Parameters
        ----------
        point_1 : array-like, shape=[n_samples, {dimension, [n, n]}]
        point_2 : array-like, shape=[n_samples, {dimension, [n, n]}]
        point_type : str, {'vector', 'matrix'}, optional
            default: default_point_type

        Returns
        -------
        point_prod : array-like, shape=[n_samples, {dimension, [n, n]}]
        """
        if point_type is None:
            point_type = self.default_point_type

        point_1 = self.regularize(point_1, point_type=point_type)
        point_2 = self.regularize(point_2, point_type=point_type)

        if point_type == 'vector':
            point_1 = self.matrix_from_rotation_vector(point_1)
            point_2 = self.matrix_from_rotation_vector(point_2)

        n_points_1 = point_1.shape[0]
        n_points_2 = point_2.shape[0]

        assert (point_1.shape == point_2.shape
                or n_points_1 == 1
                or n_points_2 == 1)

        if n_points_1 == 1:
            point_1 = gs.stack([point_1[0]] * n_points_2)

        if n_points_2 == 1:
            point_2 = gs.stack([point_2[0]] * n_points_1)

        point_prod = gs.einsum('ijk,ikl->ijl', point_1, point_2)

        if point_type == 'vector':
            point_prod = self.rotation_vector_from_matrix(point_prod)

        point_prod = self.regularize(
            point_prod, point_type=point_type)
        return point_prod

    def inverse(self, point, point_type=None):
        """Compute the group inverse in SO(n).

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dimension, [n, n]}]
        point_type : str, {'vector', 'matrix'}, optional
            default: self.default_point_type

        Returns
        -------
        inv_point : array-like, shape=[n_samples, {dimension, [n, n]}]
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            if self.n == 3:
                return -self.regularize(point, point_type=point_type)
            else:
                point = self.matrix_from_rotation_vector(point)

        transpose_order = (0, 2, 1) if gs.ndim(point) == 3 else (1, 0)
        inv_point = gs.transpose(point, transpose_order)

        if point_type == 'vector':
            inv_point = self.rotation_vector_from_matrix(inv_point)

        return inv_point

    def jacobian_translation(
            self, point, left_or_right='left', point_type=None):
        """Compute the jacobian matrix corresponding to translation.

        Compute the jacobian matrix of the differential
        of the left/right translations from the identity to point in SO(n).

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dimension, [n, n]}]
        left_or_right : str, {'left', 'right'}, optional
            default: 'left'
        point_type : str, {'vector', 'matrix'}, optional
            default: self.default_point_type

        Returns
        -------
        jacobian : array-like, shape=[n_samples, dimension, dimension]
        """
        assert left_or_right in ('left', 'right')

        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            if self.n == 3:
                point = self.regularize(
                    point, point_type=point_type)

                n_points, _ = point.shape

                angle = gs.linalg.norm(point, axis=1)
                angle = gs.expand_dims(angle, axis=1)

                coef_1 = gs.zeros([n_points, 1])
                coef_2 = gs.zeros([n_points, 1])

                # This avois dividing by 0.
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

                # This avois dividing by 0.
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

                # This avois dividing by 0.
                mask_else = ~mask_0 & ~mask_pi
                mask_else_float = gs.cast(mask_else, gs.float32) + self.epsilon

                # This avoids division by 0.
                angle += mask_pi_float
                coef_1 += mask_else_float * (
                    (angle / 2) / gs.tan(angle / 2))
                coef_2 += mask_else_float * (
                    (1 - coef_1) / angle ** 2)
                jacobian = gs.zeros((n_points, self.dimension, self.dimension))
                n_points_tensor = gs.array(n_points)
                for i in range(n_points):
                    # This avois dividing by 0.
                    mask_i_float = (
                        gs.get_mask_i_float(i, n_points_tensor)
                        + self.epsilon)

                    sign = - 1
                    if left_or_right == 'left':
                        sign = + 1

                    jacobian_i = (
                        coef_1[i] * gs.eye(self.dimension)
                        + coef_2[i] * gs.outer(point[i], point[i])
                        + sign * self.skew_matrix_from_vector(point[i]) / 2)
                    jacobian_i = gs.squeeze(jacobian_i, axis=0)

                    jacobian += gs.einsum(
                        'n,ij->nij',
                        mask_i_float,
                        jacobian_i)

            else:
                if left_or_right == 'right':
                    raise NotImplementedError(
                        'The jacobian of the right translation'
                        ' is not implemented.')
                jacobian = self.matrix_from_rotation_vector(point)

        elif point_type == 'matrix':
            raise NotImplementedError()

        return jacobian

    def random_uniform(self, n_samples=1, point_type=None):
        """Sample in SO(n) with the uniform distribution.

        Parameters
        ----------
        n_samples : int
            the amount of samples
        point_type : str, {'vector', 'matrix'}, optional
            default: self.self.default_point_type

        Returns
        -------
        point : array-like, shape=[n_samples, {dimension, [n, n]}]
        """
        if point_type is None:
            point_type = self.default_point_type

        random_point = gs.random.rand(n_samples, self.dimension) * 2 - 1
        random_point = self.regularize(random_point, point_type='vector')
        if point_type == 'matrix':
            random_point = self.matrix_from_rotation_vector(random_point)

        return random_point

    def exp_from_identity(self, tangent_vec, point_type=None):
        """Compute the group exponential of the tangent vector at the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, {dimension, [n, n]}]
        point_type : str, {'vector', 'matrix'}, optional
            default: self.default_point_type

        Returns
        -------
        point : array-like, shape=[n_samples, {dimension, [n, n]}]
        """
        if point_type is None:
            point_type = self.default_point_type

        if point_type == 'vector':
            point = gs.to_ndarray(tangent_vec, to_ndim=2)
        elif point_type == 'matrix' and self.n > 3:
            return gs.linalg.expm(tangent_vec)
        elif point_type == 'matrix':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
            tangent_vec = self.vector_from_skew_matrix(tangent_vec)
            point = self.matrix_from_rotation_vector(tangent_vec)

        return point

    def log_from_identity(self, point, point_type=None):
        """Compute the group logarithm of the point at the identity.

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dimension, [n, n]}]
        point_type : str, {'vector', 'matrix'}, optional
            default: self.default_point_type

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, {dimension, [n, n]}]
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

    def exponential_barycenter(
            self, points, weights=None, point_type=None):
        """Compute the group exponential barycenter in SO(n).

        This is the Frechet mean of the canonical bi-invariant metric on SO(n).

        Parameters
        ----------
        points : array-like, shape=[n_samples, {dimension, [n, n]}]
        weights : array-like, shape=[n_samples], optional
            default: 1 / n_samples for each point
        point_type : str, {'vector', 'matrix'}, optional
            default: self.default_point_type

        Returns
        -------
        exp_bar : array-like, shape=[{dimension, [n, n]}]
            the group exponential barycenter
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

            mean = FrechetMean(metric=self.bi_invariant_metric)
            mean.fit(X=points, weights=weights)
            exp_bar = mean.estimate_

            exp_bar = gs.to_ndarray(exp_bar, to_ndim=2)
            assert gs.ndim(exp_bar) == 2, gs.ndim(exp_bar)

        elif point_type == 'matrix':
            points = self.rotation_vector_from_matrix(points)
            exp_bar = self.exponential_barycenter(
                points, weights, point_type='vector')
            exp_bar = self.matrix_from_rotation_vector(exp_bar)

        return exp_bar
