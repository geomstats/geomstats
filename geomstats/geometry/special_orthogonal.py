"""Exposes the `SpecialOrthogonal` group class."""

import geomstats.algebra_utils as utils
import geomstats.backend as gs
import geomstats.errors
import geomstats.vectorization
from geomstats.geometry.base import EmbeddedManifold
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.invariant_metric import BiInvariantMetric
from geomstats.geometry.lie_group import LieGroup, MatrixLieGroup
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


ATOL = 1e-5

TAYLOR_COEFFS_1_AT_PI = [0., - gs.pi / 4.,
                         - 1. / 4., - gs.pi / 48.,
                         - 1. / 48., - gs.pi / 480.,
                         - 1. / 480.]


class _SpecialOrthogonalMatrices(MatrixLieGroup, EmbeddedManifold):
    """Class for special orthogonal groups in matrix representation.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        matrices = Matrices(n, n)
        gln = GeneralLinear(n, positive_det=True)
        super(_SpecialOrthogonalMatrices, self).__init__(
            dim=int((n * (n - 1)) / 2), n=n, value=gs.eye(n),
            lie_algebra=SkewSymmetricMatrices(n=n), embedding_space=gln,
            submersion=lambda x: matrices.mul(matrices.transpose(x), x),
            tangent_submersion=lambda v, x: 2 * matrices.to_symmetric(
                matrices.mul(matrices.transpose(x), v)))
        self.bi_invariant_metric = BiInvariantMetric(group=self)
        self.metric = self.bi_invariant_metric

    @classmethod
    def inverse(cls, point):
        """Return the transpose matrix of point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point in SO(n).

        Returns
        -------
        inverse : array-like, shape=[..., n, n]
            Inverse.
        """
        return Matrices.transpose(point)

    def projection(self, point):
        """Project a matrix on SO(n) by minimizing the Frobenius norm.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        rot_mat : array-like, shape=[..., n, n]
            Rotation matrix.
        """
        aux_mat = self.submersion(point)
        aux_mat = Matrices.mul(Matrices.transpose(point), point)
        inv_sqrt_mat = SymmetricMatrices.powerm(aux_mat, - 1 / 2)
        rotation_mat = Matrices.mul(point, inv_sqrt_mat)
        det = gs.linalg.det(rotation_mat)
        return utils.flip_determinant(rotation_mat, det)

    def random_point(self, n_samples=1, bound=1.):
        """Sample in SO(n) from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Unused.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled on the SO(n).
        """
        return self.random_uniform(n_samples)

    def random_uniform(self, n_samples=1):
        """Sample in SO(n) from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        tol : unused

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled on the SO(n).
        """
        if n_samples == 1:
            size = (self.n, self.n)
        else:
            size = (n_samples, self.n, self.n)
        random_mat = gs.random.normal(size=size)
        rotation_mat, _ = gs.linalg.qr(random_mat)
        det = gs.linalg.det(rotation_mat)
        return utils.flip_determinant(rotation_mat, det)

    def skew_matrix_from_vector(self, vec):
        """Get the skew-symmetric matrix derived from the vector.

        In nD, fill a skew-symmetric matrix with the values of the vector.

        Parameters
        ----------
        vec : array-like, shape=[..., dim]
            Vector.

        Returns
        -------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.
        """
        return self.lie_algebra.matrix_representation(vec)

    def vector_from_skew_matrix(self, skew_mat):
        """Derive a vector from the skew-symmetric matrix.

        In 3D, compute the vector defining the cross product
        associated to the skew-symmetric matrix skew mat.

        Parameters
        ----------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.

        Returns
        -------
        vec : array-like, shape=[..., dim]
            Vector.
        """
        return self.lie_algebra.basis_representation(skew_mat)


class _SpecialOrthogonalVectors(LieGroup):
    """Class for the special orthogonal groups SO({2,3}) in vector form.

    i.e. the Lie groups of planar and 3D rotations. This class is specific to
    the vector representation of rotations. For the matrix representation use
    the SpecialOrthogonal class and set `n=2` or `n=3`.

    Parameters
    ----------
    epsilon : float
        Precision to use for calculations involving potential divison by 0 in
        rotations.
        Optional, default: 0.
    """

    def __init__(self, n, epsilon=0.):
        dim = n * (n - 1) // 2
        LieGroup.__init__(
            self, dim=dim, default_point_type='vector')

        self.n = n
        self.epsilon = epsilon

    def get_identity(self, point_type='vector'):
        """Get the identity of the group.

        Parameters
        ----------
        point_type : str, {'vector', 'matrix'}
            Point_type of the returned value. Unused here.

        Returns
        -------
        identity : array-like, shape=[1,]
            Identity.
        """
        return gs.zeros(self.dim)

    identity = property(get_identity)

    def belongs(self, point, atol=ATOL):
        """Evaluate if a point belongs to SO(3).

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point to check whether it belongs to SO(3).
        atol : unused

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

    @geomstats.vectorization.decorator(['else', 'matrix'])
    def projection(self, point):
        """Project a matrix on SO(2) or SO(3) using the Frobenius norm.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        rot_mat : array-like, shape=[..., n, n]
            Rotation matrix.
        """
        mat = point
        n_mats, _, _ = mat.shape

        mat_unitary_u, _, mat_unitary_v = gs.linalg.svd(mat)
        rot_mat = Matrices.mul(mat_unitary_u, mat_unitary_v)
        mask = gs.less(gs.linalg.det(rot_mat), 0.)
        mask_float = gs.cast(mask, gs.float32) + self.epsilon
        diag = gs.concatenate((gs.ones(self.n - 1), -gs.ones(1)), axis=0)
        diag = gs.to_ndarray(diag, to_ndim=2)
        diag = gs.to_ndarray(
            utils.from_vector_to_diagonal_matrix(diag),
            to_ndim=3) + self.epsilon
        new_mat_diag_s = gs.tile(diag, [n_mats, 1, 1])

        aux_mat = Matrices.mul(mat_unitary_u, new_mat_diag_s)
        rot_mat = rot_mat + gs.einsum(
            '...,...jk->...jk', mask_float,
            Matrices.mul(aux_mat, mat_unitary_v))
        return rot_mat

    def inverse(self, point):
        """Compute the group inverse in SO(3).

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point.

        Returns
        -------
        inv_point : array-like, shape=[..., 3]
            Inverse.
        """
        return -self.regularize(point)

    def random_point(self, n_samples=1, bound=1.):
        return gs.random.rand(n_samples, 3)

    def exp_from_identity(self, tangent_vec):
        """Compute the group exponential of the tangent vector at the identity.

        As rotations are represented by their rotation vector,
        which corresponds to the element `X` in the Lie Algebra such that
        `exp(X) = R`, this methods returns its input without change.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dimension]
            Tangent vector at base point.

        Returns
        -------
        point : array-like, shape=[..., dimension]
            Point.
        """
        return self.regularize(tangent_vec)

    def log_from_identity(self, point):
        """Compute the group logarithm of the point at the identity.

        As rotations are represented by their rotation vector,
        which corresponds to the element `X` in the Lie Algebra such that
        `exp(X) = R`, this methods returns its input after regularization.


        Parameters
        ----------
        point : array-like, shape=[..., dimension]
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dimension]
            Group logarithm.
        """
        return self.regularize(point)

    def skew_matrix_from_vector(self, vec):
        """Get the skew-symmetric matrix derived from the vector.

        In 3D, compute the skew-symmetric matrix,known as the cross-product of
        a vector, associated to the vector `vec`.

        Parameters
        ----------
        vec : array-like, shape=[..., dim]
            Vector.

        Returns
        -------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.
        """
        return SkewSymmetricMatrices(self.n).matrix_representation(vec)

    def vector_from_skew_matrix(self, skew_mat):
        """Derive a vector from the skew-symmetric matrix.

        In 3D, compute the vector defining the cross product
        associated to the skew-symmetric matrix skew mat.

        Parameters
        ----------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.

        Returns
        -------
        vec : array-like, shape=[..., dim]
            Vector.
        """
        return SkewSymmetricMatrices(self.n).basis_representation(skew_mat)

    def to_tangent(self, vector, base_point=None):
        return self.regularize_tangent_vec(vector, base_point)

    def regularize_tangent_vec_at_identity(
            self, tangent_vec, metric=None):
        """Regularize a tangent vector at the identity.

        In 2D, regularize a tangent_vector by getting its norm at the identity,
        to be less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        metric : RiemannianMetric
            Metric to compute the norm of the tangent vector.
            Optional, default is the Euclidean metric.

        Returns
        -------
        regularized_vec : array-like, shape=[..., 1]
            Regularized tangent vector.
        """
        return self.regularize(tangent_vec)

    def regularize_tangent_vec(
            self, tangent_vec, base_point, metric=None):
        """Regularize tangent vector at a base point.

        In 2D, regularize a tangent_vector by getting the norm of its parallel
        transport to the identity, determined by the metric, less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[...,1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Point on the manifold.
        metric : RiemannianMetric
            Metric to compute the norm of the tangent vector.
            Optional, default is the Euclidean metric.

        Returns
        -------
        regularized_tangent_vec : array-like, shape=[..., 1]
            Regularized tangent vector.
        """
        return self.regularize_tangent_vec_at_identity(tangent_vec)


class _SpecialOrthogonal2Vectors(_SpecialOrthogonalVectors):
    """Class for the special orthogonal group SO(2) in vector representation.

    i.e. the Lie group of planar rotations. This class is specific to the
    vector representation of rotations. For the matrix representation use the
    SpecialOrthogonal class and set `n=2`.

    Parameters
    ----------
    epsilon : float
        Precision to use for calculations involving potential divison by 0 in
        rotations.
        Optional, default: 0.
    """

    def __init__(self, epsilon=0.):
        super(_SpecialOrthogonal2Vectors, self).__init__(
            n=2, epsilon=epsilon)

    def regularize(self, point):
        """Regularize a point to be in accordance with convention.

        In 2D, regularize the norm of the rotation angle,
        to be between -pi and pi, following the axis-angle
        representation's convention.

        If the angle angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        Parameters
        ----------
        point : array-like, shape=[...,1]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., 1]
            Regularized point.
        """
        regularized_point = point
        regularized_point = gs.mod(regularized_point, 2 * gs.pi)
        regularized_point = gs.where(
            regularized_point < gs.pi,
            regularized_point, regularized_point - 2 * gs.pi)
        return regularized_point

    def rotation_vector_from_matrix(self, rot_mat):
        r"""Convert rotation matrix (in 2D) to rotation vector (axis-angle).

        Get the angle through the atan2 function:

        Parameters
        ----------
        rot_mat : array-like, shape=[..., 2, 2]
            Rotation matrix.

        Returns
        -------
        regularized_rot_vec : array-like, shape=[..., 1]
            Rotation vector.
        """
        rot_vec = gs.arctan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])
        return self.regularize(rot_vec[..., None])

    def matrix_from_rotation_vector(self, rot_vec):
        """Convert rotation vector to rotation matrix.

        Parameters
        ----------
        rot_vec: array-like, shape=[..., 1]
            Rotation vector.

        Returns
        -------
        rot_mat: array-like, shape=[..., 2, 2]
            Rotation matrix.
        """
        rot_vec = self.regularize(rot_vec)

        cos_term = gs.cos(rot_vec)
        cos_matrix = gs.einsum('...l,ij->...ij', cos_term, gs.eye(2))
        sin_term = gs.sin(rot_vec)
        sin_matrix = self.skew_matrix_from_vector(sin_term)
        return cos_matrix + sin_matrix

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

        point_prod = point_a + point_b
        point_prod = self.regularize(point_prod)

        return point_prod

    def random_point(self, n_samples=1, bound=1.):
        return self.random_uniform(n_samples)

    def random_uniform(self, n_samples=1):
        """Sample in SO(2) with the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        point : array-like, shape=[..., 1]
            Sample.
        """
        random_point = (gs.random.rand(n_samples, self.dim) * 2 - 1) * gs.pi
        random_point = self.regularize(random_point)

        if n_samples == 1:
            random_point = gs.squeeze(random_point, axis=0)

        return random_point

    def exp(self, tangent_vec, base_point=None):
        """Compute the group exponential.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Point from which the exponential is computed.

        Returns
        -------
        point : array-like, shape=[..., 1]
            Group exponential.
        """
        return self.regularize(tangent_vec + base_point)

    def log(self, point, base_point=None):
        """Compute the group logarithm.

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point.
        base_point : array-like, shape=[..., 1]
            Point from which the log is computed.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Group logarithm.
        """
        return self.regularize(point - base_point)


class _SpecialOrthogonal3Vectors(_SpecialOrthogonalVectors):
    """Class for the special orthogonal group SO(3) in vector representation.

    i.e. the Lie group of rotations. This class is specific to the vector
    representation of rotations. For the matrix representation use the
    SpecialOrthogonal class and set `n=3`.

    Parameters
    ----------
    epsilon : float
        Precision to use for calculations involving potential divison by 0 in
        rotations.
        Optional, default: 0.
    """

    def __init__(self, epsilon=0.):
        super(_SpecialOrthogonal3Vectors, self).__init__(
            n=3, epsilon=epsilon)

        self.bi_invariant_metric = BiInvariantMetric(group=self)
        self.metric = self.bi_invariant_metric

    def regularize(self, point):
        """Regularize a point to be in accordance with convention.

        In 3D, regularize the norm of the rotation vector,
        to be between 0 and pi, following the axis-angle
        representation's convention.

        If the angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        Parameters
        ----------
        point : array-like, shape=[...,3]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., 3]
            Regularized point.
        """
        theta = gs.linalg.norm(point, axis=-1)
        k = gs.floor(theta / 2. / gs.pi)

        # angle in [0;2pi)
        angle = theta - 2 * k * gs.pi

        # this avoids dividing by 0
        theta_eps = gs.where(gs.isclose(theta, 0.), 1., theta)

        # angle in [0, pi]
        normalized_angle = gs.where(angle <= gs.pi, angle, 2 * gs.pi - angle)
        norm_ratio = gs.where(
            gs.isclose(theta, 0.), 1., normalized_angle / theta_eps)

        # reverse sign if angle was greater than pi
        norm_ratio = gs.where(angle > gs.pi, -norm_ratio, norm_ratio)
        return gs.einsum(
            '...,...i->...i', norm_ratio, point)

    def regularize_tangent_vec_at_identity(
            self, tangent_vec, metric=None):
        """Regularize a tangent vector at the identity.

        In 3D, regularize a tangent_vector by getting its norm at the identity,
        determined by the metric, to be less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 3]
            Tangent vector at base point.
        metric : RiemannianMetric
            Metric.
            Optional, default: self.left_canonical_metric.

        Returns
        -------
        regularized_vec : array-like, shape=[..., 3]
            Regularized tangent vector.
        """
        if metric is None:
            return self.regularize(tangent_vec)

        tangent_vec_metric_norm = metric.norm(tangent_vec)
        tangent_vec_canonical_norm = gs.linalg.norm(tangent_vec, axis=-1)

        # This avoids dividing by 0
        norm_eps = gs.where(
            tangent_vec_canonical_norm == 0,
            gs.atol, tangent_vec_canonical_norm)
        coef = gs.where(
            tangent_vec_canonical_norm == 0.,
            1., tangent_vec_metric_norm / norm_eps)
        coef_tangent_vec = gs.einsum('...,...i->...i', coef, tangent_vec)

        regularized_vec = self.regularize(coef_tangent_vec)
        return gs.einsum(
            '...,...i->...i', 1. / coef, regularized_vec)

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
        metric : RiemannianMetric
            Metric.
            Optional, default: self.left_canonical_metric.

        Returns
        -------
        regularized_tangent_vec : array-like, shape=[..., 3]
            Regularized tangent vector.
        """
        if metric is None:
            metric = self.left_canonical_metric
        base_point = self.regularize(base_point)

        tangent_vec_at_id = self.tangent_translation_map(
            base_point, left_or_right=metric.left_or_right, inverse=True)(
            tangent_vec
        )

        tangent_vec_at_id = self.regularize_tangent_vec_at_identity(
            tangent_vec_at_id, metric)

        regularized_tangent_vec = self.tangent_translation_map(
            base_point, left_or_right=metric.left_or_right)(tangent_vec_at_id)

        return regularized_tangent_vec

    @geomstats.vectorization.decorator(['else', 'matrix', 'output_point'])
    def rotation_vector_from_matrix(self, rot_mat):
        r"""Convert rotation matrix (in 3D) to rotation vector (axis-angle).

        Get the angle through the trace of the rotation matrix:
        The eigenvalues are:
        :math:`\{1, \cos(angle) + i \sin(angle), \cos(angle) - i \sin(angle)\}`
        so that:
        :math:`trace = 1 + 2 \cos(angle), \{-1 \leq trace \leq 3\}`

        The rotation vector is the vector associated to the skew-symmetric
        matrix
        :math:`S_r = \frac{angle}{(2 * \sin(angle) ) (R - R^T)}`

        For the edge case where the angle is close to pi,
        the rotation vector (up to sign) is derived by using the following
        equality (see the Axis-angle representation on Wikipedia):
        :math:`outer(r, r) = \frac{1}{2} (R + I_3)`
        In nD, the rotation vector stores the :math:`n(n-1)/2` values
        of the skew-symmetric matrix representing the rotation.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., n, n]
            Rotation matrix.

        Returns
        -------
        regularized_rot_vec : array-like, shape=[..., 3]
            Rotation vector.
        """
        n_rot_mats, _, _ = rot_mat.shape

        trace = gs.trace(rot_mat, axis1=1, axis2=2)
        trace = gs.to_ndarray(trace, to_ndim=2, axis=1)
        trace_num = gs.clip(trace, -1, 3)
        angle = gs.arccos(0.5 * (trace_num - 1))
        rot_mat_transpose = gs.transpose(rot_mat, axes=(0, 2, 1))
        rot_vec_not_pi = self.vector_from_skew_matrix(
            rot_mat - rot_mat_transpose)
        mask_0 = gs.cast(gs.isclose(angle, 0.), angle.dtype)
        mask_pi = gs.cast(gs.isclose(angle, gs.pi, atol=1e-2), angle.dtype)
        mask_else = (1 - mask_0) * (1 - mask_pi)

        numerator = 0.5 * mask_0 + angle * mask_else
        denominator = (1 - angle ** 2 / 6) * mask_0 + 2 * gs.sin(
            angle) * mask_else + mask_pi

        rot_vec_not_pi = rot_vec_not_pi * numerator / denominator

        vector_outer = 0.5 * (gs.eye(3) + rot_mat)
        vector_outer = gs.set_diag(
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

    def matrix_from_rotation_vector(self, rot_vec):
        """Convert rotation vector to rotation matrix.

        Parameters
        ----------
        rot_vec: array-like, shape=[..., 3]
            Rotation vector.

        Returns
        -------
        rot_mat: array-like, shape=[..., 3]
            Rotation matrix.
        """
        rot_vec = self.regularize(rot_vec)

        squared_angle = gs.sum(rot_vec ** 2, axis=-1)
        skew_rot_vec = self.skew_matrix_from_vector(rot_vec)

        coef_1 = utils.taylor_exp_even_func(squared_angle, utils.sinc_close_0)
        coef_2 = utils.taylor_exp_even_func(squared_angle, utils.cosc_close_0)

        term_1 = (gs.eye(self.dim)
                  + gs.einsum('...,...jk->...jk', coef_1, skew_rot_vec))

        squared_skew_rot_vec = Matrices.mul(skew_rot_vec, skew_rot_vec)

        term_2 = gs.einsum('...,...jk->...jk', coef_2, squared_skew_rot_vec)

        return term_1 + term_2

    @geomstats.vectorization.decorator(['else', 'matrix'])
    def quaternion_from_matrix(self, rot_mat):
        """Convert a rotation matrix into a unit quaternion.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., 3, 3]
            Rotation matrix.

        Returns
        -------
        quaternion : array-like, shape=[..., 4]
            Quaternion.
        """
        rot_vec = self.rotation_vector_from_matrix(rot_mat)
        quaternion = self.quaternion_from_rotation_vector(rot_vec)

        return quaternion

    def quaternion_from_rotation_vector(self, rot_vec):
        """Convert a rotation vector into a unit quaternion.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]
            Rotation vector.

        Returns
        -------
        quaternion : array-like, shape=[..., 4]
            Quaternion.
        """
        rot_vec = self.regularize(rot_vec)

        squared_angle = gs.sum(rot_vec ** 2, axis=-1)

        coef_cos = utils.taylor_exp_even_func(
            squared_angle / 4, utils.cos_close_0)
        coef_sinc = .5 * utils.taylor_exp_even_func(
            squared_angle / 4, utils.sinc_close_0)

        quaternion = gs.concatenate((
            coef_cos[..., None],
            gs.einsum('...,...i->...i', coef_sinc, rot_vec)), axis=-1)

        return quaternion

    def rotation_vector_from_quaternion(self, quaternion):
        """Convert a unit quaternion into a rotation vector.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]
            Quaternion.

        Returns
        -------
        rot_vec : array-like, shape=[..., 3]
            Rotation vector.
        """
        cos_half_angle = quaternion[..., 0]
        cos_half_angle = gs.clip(cos_half_angle, -1, 1)
        half_angle = gs.arccos(cos_half_angle)

        coef_isinc = 2 * utils.taylor_exp_even_func(
            half_angle ** 2, utils.inv_sinc_close_0)

        rot_vec = gs.einsum(
            '...,...i->...i', coef_isinc, quaternion[..., 1:])

        rot_vec = self.regularize(rot_vec)
        return rot_vec

    @geomstats.vectorization.decorator(['else', 'vector'])
    def matrix_from_quaternion(self, quaternion):
        """Convert a unit quaternion into a rotation vector.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]
            Quaternion.

        Returns
        -------
        rot_mat : array-like, shape=[..., 3]
            Rotation matrix.
        """
        n_quaternions, _ = quaternion.shape

        w, x, y, z = gs.hsplit(quaternion, 4)

        rot_mat = gs.zeros((n_quaternions,) + (self.n,) * 2)

        for i in range(n_quaternions):
            # TODO (nina): Vectorize by applying the composition of
            # quaternions to the identity matrix
            column_1 = gs.array([
                w[i] ** 2 + x[i] ** 2 - y[i] ** 2 - z[i] ** 2,
                2 * x[i] * y[i] - 2 * w[i] * z[i],
                2 * x[i] * z[i] + 2 * w[i] * y[i]])

            column_2 = gs.array([
                2 * x[i] * y[i] + 2 * w[i] * z[i],
                w[i] ** 2 - x[i] ** 2 + y[i] ** 2 - z[i] ** 2,
                2 * y[i] * z[i] - 2 * w[i] * x[i]])

            column_3 = gs.array([
                2 * x[i] * z[i] - 2 * w[i] * y[i],
                2 * y[i] * z[i] + 2 * w[i] * x[i],
                w[i] ** 2 - x[i] ** 2 - y[i] ** 2 + z[i] ** 2])

            mask_i = gs.get_mask_i_float(i, n_quaternions)
            rot_mat_i = gs.transpose(
                gs.hstack([column_1, column_2, column_3]))
            rot_mat_i = gs.to_ndarray(rot_mat_i, to_ndim=3)
            rot_mat += gs.einsum('...,...ij->...ij', mask_i, rot_mat_i)

        return rot_mat

    @staticmethod
    @geomstats.vectorization.decorator(['vector'])
    def matrix_from_tait_bryan_angles_extrinsic_xyz(tait_bryan_angles):
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

        rot_mat = []
        angle_1 = tait_bryan_angles[:, 0]
        angle_2 = tait_bryan_angles[:, 1]
        angle_3 = tait_bryan_angles[:, 2]

        # TODO: avoid for loop in vectorization of tait bryan angles
        for i in range(n_tait_bryan_angles):
            cos_angle_1 = gs.cos(angle_1[i])
            sin_angle_1 = gs.sin(angle_1[i])
            cos_angle_2 = gs.cos(angle_2[i])
            sin_angle_2 = gs.sin(angle_2[i])
            cos_angle_3 = gs.cos(angle_3[i])
            sin_angle_3 = gs.sin(angle_3[i])

            column_1 = gs.array([
                [cos_angle_1 * cos_angle_2],
                [cos_angle_2 * sin_angle_1],
                [- sin_angle_2]])
            column_2 = gs.array([
                [(cos_angle_1 * sin_angle_2 * sin_angle_3
                  - cos_angle_3 * sin_angle_1)],
                [(cos_angle_1 * cos_angle_3
                  + sin_angle_1 * sin_angle_2 * sin_angle_3)],
                [cos_angle_2 * sin_angle_3]])
            column_3 = gs.array([
                [(sin_angle_1 * sin_angle_3
                  + cos_angle_1 * cos_angle_3 * sin_angle_2)],
                [(cos_angle_3 * sin_angle_1 * sin_angle_2
                  - cos_angle_1 * sin_angle_3)],
                [cos_angle_2 * cos_angle_3]])

            rot_mat.append(gs.hstack((column_1, column_2, column_3)))
        return gs.stack(rot_mat)

    @staticmethod
    @geomstats.vectorization.decorator(['vector'])
    def matrix_from_tait_bryan_angles_extrinsic_zyx(tait_bryan_angles):
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

        rot_mat = []
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

            column_1 = gs.array(
                [[cos_angle_2 * cos_angle_3],
                 [(cos_angle_1 * sin_angle_3
                   + cos_angle_3 * sin_angle_1 * sin_angle_2)],
                 [(sin_angle_1 * sin_angle_3
                   - cos_angle_1 * cos_angle_3 * sin_angle_2)]])

            column_2 = gs.array(
                [[- cos_angle_2 * sin_angle_3],
                 [(cos_angle_1 * cos_angle_3
                   - sin_angle_1 * sin_angle_2 * sin_angle_3)],
                 [(cos_angle_3 * sin_angle_1
                   + cos_angle_1 * sin_angle_2 * sin_angle_3)]])

            column_3 = gs.array(
                [[sin_angle_2],
                 [- cos_angle_2 * sin_angle_1],
                 [cos_angle_1 * cos_angle_2]])
            rot_mat.append(gs.hstack((column_1, column_2, column_3)))
        return gs.stack(rot_mat)

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

    def jacobian_translation(
            self, point, left_or_right='left'):
        """Compute the jacobian matrix corresponding to translation.

        Compute the jacobian matrix of the differential
        of the left/right translations from the identity to point in SO(3).

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point.
        left_or_right : str, {'left', 'right'}
            Whether to use left or right invariant metric.
            Optional, default: 'left'.

        Returns
        -------
        jacobian : array-like, shape=[..., 3, 3]
            Jacobian.
        """
        geomstats.errors.check_parameter_accepted_values(
            left_or_right, 'left_or_right', ['left', 'right'])

        point = self.regularize(point)
        squared_angle = gs.sum(point ** 2, axis=-1)

        angle = gs.sqrt(squared_angle)
        delta_angle = angle - gs.pi
        approx_at_pi = gs.sum(gs.array([
            TAYLOR_COEFFS_1_AT_PI[k] * delta_angle ** k for k in range(1, 7)
        ]))
        coef_1 = utils.taylor_exp_even_func(
            squared_angle / 4, utils.inv_tanc_close_0)
        coef_1 = gs.where(
            -delta_angle < utils.EPSILON, approx_at_pi, coef_1)

        coef_2 = utils.taylor_exp_even_func(
            squared_angle, utils.var_inv_tanc_close_0, order=4)
        squared_angle_ = gs.where(
            squared_angle < utils.EPSILON, utils.EPSILON, squared_angle)
        coef_2 = gs.where(
            squared_angle < utils.EPSILON,
            coef_2, (1 - coef_1) / squared_angle_)

        outer_ = gs.einsum('...i,...j->...ij', point, point)
        sign = - 1. if left_or_right == 'right' else 1.

        return (
            gs.einsum('...,...ij->...ij', coef_1, gs.eye(self.dim))
            + gs.einsum('...,...ij->...ij', coef_2, outer_)
            + sign * self.skew_matrix_from_vector(point) / 2.)

    def random_uniform(self, n_samples=1):
        """Sample in SO(3) with the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        point : array-like, shape=[..., 3]
            Sample.
        """
        random_point = gs.random.rand(n_samples, self.dim) * 2 - 1
        random_point = self.regularize(random_point)

        if n_samples == 1:
            random_point = gs.squeeze(random_point, axis=0)

        return random_point

    def lie_bracket(
            self, tangent_vector_a, tangent_vector_b, base_point=None):
        """Compute the lie bracket of two tangent vectors.

        For matrix Lie groups with tangent vectors A,B at the same base point P
        this is given by (translate to identity, compute commutator, go back)
        :math:`[A,B] = A_P^{-1}B - B_P^{-1}A`

        Parameters
        ----------
        tangent_vector_a : shape=[..., n, n]
            Tangent vector at base point.
        tangent_vector_b : shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.
            Optional, default: None.

        Returns
        -------
        bracket : array-like, shape=[..., n, n]
            Lie bracket.
        """
        return gs.cross(tangent_vector_a, tangent_vector_b)

    def exp(self, tangent_vec, base_point=None):
        """Compute the group exponential.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 3]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 3]
            Group element.

        Returns
        -------
        point : array-like, shape=[..., 3]
            Group exponential.
        """
        return LieGroup.exp(self, tangent_vec, base_point)

    def log(self, point, base_point=None):
        """Compute the group logarithm.

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point of the group, i.e. rotation vector.
        base_point : array-like, shape=[..., 3]
            Base point for the log, i.e. rotation vector.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 3]
            Group logarithm.
        """
        return LieGroup.log(self, point, base_point)


class SpecialOrthogonal(_SpecialOrthogonal2Vectors,
                        _SpecialOrthogonal3Vectors,
                        _SpecialOrthogonalMatrices):
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
        if n == 2 and point_type == 'vector':
            return _SpecialOrthogonal2Vectors(epsilon)
        if n == 3 and point_type == 'vector':
            return _SpecialOrthogonal3Vectors(epsilon)
        if point_type == 'vector':
            raise NotImplementedError(
                'SO(n) is only implemented in vector representation'
                ' when n = 3.')
        return _SpecialOrthogonalMatrices(n)
