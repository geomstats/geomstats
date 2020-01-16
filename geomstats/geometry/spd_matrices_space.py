"""
The manifold of symmetric positive definite (SPD) matrices.
"""

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.general_linear_group import GeneralLinearGroup
from geomstats.geometry.riemannian_metric import RiemannianMetric

EPSILON = 1e-6
TOLERANCE = 1e-12


class SPDMatricesSpace(EmbeddedManifold):
    """
    Class for the manifold of symmetric positive definite (SPD) matrices.
    """
    def __init__(self, n):
        assert isinstance(n, int) and n > 0
        super(SPDMatricesSpace, self).__init__(
            dimension=int(n * (n + 1) / 2),
            embedding_manifold=GeneralLinearGroup(n=n))
        self.n = n

    def belongs(self, mat, tolerance=TOLERANCE):
        """
        Check if a matrix belongs to the manifold of
        symmetric positive definite matrices.
        """
        mat = gs.to_ndarray(mat, to_ndim=3)
        n_mats, mat_dim, _ = mat.shape

        mask_is_symmetric = self.embedding_manifold.is_symmetric(
                mat, tolerance=tolerance)
        mask_is_invertible = self.embedding_manifold.belongs(mat)

        belongs = mask_is_symmetric & mask_is_invertible
        belongs = gs.to_ndarray(belongs, to_ndim=1)
        belongs = gs.to_ndarray(belongs, to_ndim=2, axis=1)
        return belongs

    def vector_from_symmetric_matrix(self, mat):
        """
        Convert the symmetric part of a symmetric matrix
        into a vector.
        """
        mat = gs.to_ndarray(mat, to_ndim=3)
        assert gs.all(self.embedding_manifold.is_symmetric(mat))
        mat = self.embedding_manifold.make_symmetric(mat)

        _, mat_dim, _ = mat.shape
        vec_dim = int(mat_dim * (mat_dim + 1) / 2)
        vec = gs.zeros(vec_dim)

        idx = 0
        for i in range(mat_dim):
            for j in range(i + 1):
                if i == j:
                    vec[idx] = mat[j, j]
                else:
                    vec[idx] = mat[j, i]
                idx += 1

        return vec

    def symmetric_matrix_from_vector(self, vec):
        """
        Convert a vector into a symmetric matrix.
        """
        vec = gs.to_ndarray(vec, to_ndim=2)
        _, vec_dim = vec.shape
        mat_dim = int((gs.sqrt(8 * vec_dim + 1) - 1) / 2)
        mat = gs.zeros((mat_dim,) * 2)

        lower_triangle_indices = gs.tril_indices(mat_dim)
        diag_indices = gs.diag_indices(mat_dim)

        mat[lower_triangle_indices] = 2 * vec
        mat[diag_indices] = vec

        mat = self.embedding_manifold.make_symmetric(mat)
        return mat

    def random_uniform(self, n_samples=1):
        mat = 2 * gs.random.rand(n_samples, self.n, self.n) - 1

        spd_mat = self.embedding_manifold.group_exp(
                mat + gs.transpose(mat, axes=(0, 2, 1)))
        return spd_mat

    def random_tangent_vec_uniform(self, n_samples=1, base_point=None):
        if base_point is None:
            base_point = gs.eye(self.n)

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        assert n_base_points == n_samples or n_base_points == 1
        if n_base_points == 1:
            base_point = gs.tile(base_point, (n_samples, 1, 1))

        sqrt_base_point = gs.linalg.sqrtm(base_point)

        tangent_vec_at_id = (2 * gs.random.rand(n_samples,
                                                self.n,
                                                self.n)
                             - 1)
        tangent_vec_at_id = (tangent_vec_at_id
                             + gs.transpose(tangent_vec_at_id,
                                            axes=(0, 2, 1)))

        tangent_vec = gs.matmul(sqrt_base_point, tangent_vec_at_id)
        tangent_vec = gs.matmul(tangent_vec, sqrt_base_point)

        return tangent_vec

    def aux_differential_power(self, power, tangent_vec, base_point):
        """
        Auxiliary function to the functions differential_power and
        inverse_differential_power.

        Parameters
        ----------
        power : int
        tangent_vec : array_like, shape=[n_samples, n, n]
                      Tangent vectors.
        base_point : array_like, shape=[n_samples, n, n]
                     Base points.

        Returns
        -------
        eigvectors : array-like, shape=[n_samples, n, n]
        transp_eigvectors : array-like, shape=[n_samples, n, n]
        numerator : array-like, shape=[n_samples, n, n]
        denominator : array-like, shape=[n_samples, n, n]
        temp_result : array-like, shape=[n_samples, n, n]
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_tangent_vecs, _, _ = tangent_vec.shape
        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        assert (n_tangent_vecs == n_base_points
                or n_base_points == 1
                or n_tangent_vecs == 1)

        eigvalues, eigvectors = gs.linalg.eigh(base_point)
        eigvalues = gs.to_ndarray(eigvalues, to_ndim=3, axis=1)
        transp_eigvalues = gs.transpose(eigvalues, (0, 2, 1))
        powered_eigvalues = eigvalues**power
        transp_powered_eigvalues = gs.transpose(powered_eigvalues, (0, 2, 1))
        ones = gs.ones((n_base_points, 1, self.n))
        transp_ones = gs.transpose(ones, (0, 2, 1))

        vertical_index = gs.matmul(transp_eigvalues, ones)
        horizontal_index = gs.matmul(transp_ones, eigvalues)
        vertical_index_power = gs.matmul(transp_powered_eigvalues, ones)
        horizontal_index_power = gs.matmul(transp_ones, powered_eigvalues)
        denominator = vertical_index - horizontal_index
        numerator = vertical_index_power - horizontal_index_power
        numerator = gs.where(denominator == 0, power*vertical_index_power,
                             numerator)
        denominator = gs.where(denominator == 0, vertical_index, denominator)

        transp_eigvectors = gs.transpose(eigvectors, (0, 2, 1))
        temp_result = gs.matmul(transp_eigvectors, tangent_vec)
        temp_result = gs.matmul(temp_result, eigvectors)
        return (eigvectors, transp_eigvectors, numerator, denominator,
                temp_result)

    def differential_power(self, power, tangent_vec, base_point):
        """
        Computes the differential of the power function on SPD
        matrices (A^p=exp(p log(A))) at base_point applied to
        tangent_vec.

        Parameters
        ----------
        power : int
        tangent_vec : array_like, shape=[n_samples, n, n]
                      Tangent vectors.
        base_point : array_like, shape=[n_samples, n, n]
                     Base points.

        Returns
        -------
        differential_power : array-like, shape=[n_samples, n, n]
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result =\
            self.aux_differential_power(power, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result

    def inverse_differential_power(self, power, tangent_vec, base_point):
        """
        Computes the inverse of the differential of the power
        function on SPD matrices (A^p=exp(p log(A))) at base_point
        applied to tangent_vec.

        Parameters
        ----------
        power : int
        tangent_vec : array_like, shape=[n_samples, n, n]
                      Tangent vectors.
        base_point : array_like, shape=[n_samples, n, n]
                     Base points.

        Returns
        -------
        inverse_differential_power : array-like, shape=[n_samples, n, n]
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result =\
            self.aux_differential_power(power, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result


class SPDMetricAffine(RiemannianMetric):

    def __init__(self, n, power_affine=1):
        """
        Parameters
        ----------
        n : int
            Matrix dimension.
        power_affine : int, optional
                       Power transformation of the classical SPD metric.
        Based on:
        Thanwerdas, Pennec
        "Is affine-invariance well defined on SPD matrices?
        A principled continuum of metrics"
        Proc. of GSI 2019

        https://arxiv.org/abs/1906.01349
        """
        dimension = int(n * (n + 1) / 2)
        super(SPDMetricAffine, self).__init__(
                dimension=dimension,
                signature=(dimension, 0, 0))
        self.n = n
        self.space = SPDMatricesSpace(n)
        self.power_affine = power_affine

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """
        Compute the inner product of tangent_vec_a and tangent_vec_b
        at point base_point using the affine invariant Riemannian metric.
        """
        power_affine = self.power_affine
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_a, _, _ = tangent_vec_a.shape
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
        n_tangent_vecs_b, _, _ = tangent_vec_b.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        spd_space = self.space

        assert (n_tangent_vecs_a == n_tangent_vecs_b == n_base_points
                or n_tangent_vecs_a == n_tangent_vecs_b and n_base_points == 1
                or n_base_points == n_tangent_vecs_a and n_tangent_vecs_b == 1
                or n_base_points == n_tangent_vecs_b and n_tangent_vecs_a == 1
                or n_tangent_vecs_a == 1 and n_tangent_vecs_b == 1
                or n_base_points == 1 and n_tangent_vecs_a == 1
                or n_base_points == 1 and n_tangent_vecs_b == 1)

        if n_tangent_vecs_a == 1:
            tangent_vec_a = gs.tile(
                tangent_vec_a,
                (gs.maximum(n_base_points, n_tangent_vecs_b), 1, 1))

        if n_tangent_vecs_b == 1:
            tangent_vec_b = gs.tile(
                tangent_vec_b,
                (gs.maximum(n_base_points, n_tangent_vecs_a), 1, 1))

        if n_base_points == 1:
            base_point = gs.tile(
                base_point,
                (gs.maximum(n_tangent_vecs_a, n_tangent_vecs_b), 1, 1))

        inv_base_point = gs.linalg.inv(base_point)

        if power_affine == 1:
            aux_a = gs.matmul(inv_base_point, tangent_vec_a)
            aux_b = gs.matmul(inv_base_point, tangent_vec_b)
            inner_product = gs.trace(gs.matmul(aux_a, aux_b), axis1=1, axis2=2)
        else:
            modified_tangent_vec_a =\
                spd_space.differential_power(power_affine, tangent_vec_a,
                                             base_point)
            modified_tangent_vec_b =\
                spd_space.differential_power(power_affine, tangent_vec_b,
                                             base_point)
            power_log_inv_base_point =\
                power_affine * gs.linalg.logm(inv_base_point)
            power_inv_base_point = gs.linalg.expm(power_log_inv_base_point)
            aux_a = gs.matmul(power_inv_base_point, modified_tangent_vec_a)
            aux_b = gs.matmul(power_inv_base_point, modified_tangent_vec_b)
            product = gs.matmul(aux_a, aux_b)
            inner_product = gs.trace(product, axis1=1, axis2=2)\
                / (power_affine**2)

        inner_product = gs.to_ndarray(inner_product, to_ndim=2, axis=1)

        return inner_product

    def exp(self, tangent_vec, base_point):
        """
        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric
        defined in inner_product.

        This gives a symmetric positive definite matrix.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_tangent_vecs, _, _ = tangent_vec.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, mat_dim, _ = base_point.shape

        assert (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1)

        if n_tangent_vecs == 1:
            tangent_vec = gs.tile(tangent_vec, (n_base_points, 1, 1))
        if n_base_points == 1:
            base_point = gs.tile(base_point, (n_tangent_vecs, 1, 1))

        sqrt_base_point = gs.linalg.sqrtm(base_point)

        inv_sqrt_base_point = gs.linalg.inv(sqrt_base_point)

        tangent_vec_at_id = gs.matmul(inv_sqrt_base_point,
                                      tangent_vec)
        tangent_vec_at_id = gs.matmul(tangent_vec_at_id,
                                      inv_sqrt_base_point)
        exp_from_id = gs.linalg.expm(tangent_vec_at_id)

        exp = gs.matmul(exp_from_id, sqrt_base_point)
        exp = gs.matmul(sqrt_base_point, exp)

        return exp

    def log(self, point, base_point):
        """
        Compute the Riemannian logarithm at point base_point,
        of point wrt the metric defined in inner_product.

        This gives a tangent vector at point base_point.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        n_points, _, _ = point.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, mat_dim, _ = base_point.shape

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        if n_points == 1:
            point = gs.tile(point, (n_base_points, 1, 1))
        if n_base_points == 1:
            base_point = gs.tile(base_point, (n_points, 1, 1))

        sqrt_base_point = gs.zeros((n_base_points,) + (mat_dim,) * 2)
        sqrt_base_point = gs.linalg.sqrtm(base_point)

        inv_sqrt_base_point = gs.linalg.inv(sqrt_base_point)
        point_near_id = gs.matmul(inv_sqrt_base_point, point)
        point_near_id = gs.matmul(point_near_id, inv_sqrt_base_point)
        log_at_id = gs.linalg.logm(point_near_id)

        log = gs.matmul(sqrt_base_point, log_at_id)
        log = gs.matmul(log, sqrt_base_point)

        return log

    def geodesic(self, initial_point, initial_tangent_vec):
        return super(SPDMetricAffine, self).geodesic(
                                      initial_point=initial_point,
                                      initial_tangent_vec=initial_tangent_vec,
                                      point_type='matrix')


class SPDMetricProcrustes(RiemannianMetric):

    """
    Class for the Procrustes/Bures-Wasserstein metric on
    the manifold of SPD matrices.

    Based on :
    Bhatia, Jain, Lim
    "On the Bures-Wasserstein distance between positive
    definite matrices"
    Elsevier, Expositiones Mathematicae, vol. 37(2), 165-191

    https://arxiv.org/pdf/1712.01504.pdf
    """

    def __init__(self, n):
        dimension = int(n * (n + 1) / 2)
        super(SPDMetricProcrustes, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
        self.n = n
        self.space = SPDMatricesSpace(n)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """
        Compute the inner product of tangent_vec_a and tangent_vec_b
        at point base_point using the Procrustes Riemannian metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, n, n]
        tangent_vec_b : array-like, shape=[n_samples, n, n]
        base_point : array-like, shape={n_samples, n, n]

        Returns
        -------
        inner_product : float
        """
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_a, _, _ = tangent_vec_a.shape
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
        n_tangent_vecs_b, _, _ = tangent_vec_b.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        assert (n_tangent_vecs_a == n_tangent_vecs_b == n_base_points
                or n_tangent_vecs_a == n_tangent_vecs_b and n_base_points == 1
                or n_base_points == n_tangent_vecs_a and n_tangent_vecs_b == 1
                or n_base_points == n_tangent_vecs_b and n_tangent_vecs_a == 1
                or n_tangent_vecs_a == 1 and n_tangent_vecs_b == 1
                or n_base_points == 1 and n_tangent_vecs_a == 1
                or n_base_points == 1 and n_tangent_vecs_b == 1)

        if n_tangent_vecs_a == 1:
            tangent_vec_a = gs.tile(
                tangent_vec_a,
                (gs.maximum(n_base_points, n_tangent_vecs_b), 1, 1))

        if n_tangent_vecs_b == 1:
            tangent_vec_b = gs.tile(
                tangent_vec_b,
                (gs.maximum(n_base_points, n_tangent_vecs_a), 1, 1))

        if n_base_points == 1:
            base_point = gs.tile(
                base_point,
                (gs.maximum(n_tangent_vecs_a, n_tangent_vecs_b), 1, 1))

        spd_space = self.space
        modified_tangent_vec_a =\
            spd_space.inverse_differential_power(2, tangent_vec_a, base_point)
        product = gs.matmul(modified_tangent_vec_a, tangent_vec_b)
        result = gs.trace(product, axis1=1, axis2=2) / 2
        return result


class SPDMetricEuclidean(RiemannianMetric):

    def __init__(self, n, power_euclidean=1):
        dimension = int(n * (n + 1) / 2)
        super(SPDMetricEuclidean, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
        self.n = n
        self.space = SPDMatricesSpace(n)
        self.power_euclidean = power_euclidean

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """
        Compute the inner product of tangent_vec_a and tangent_vec_b
        at point base_point using the power-Euclidean metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, n, n]
        tangent_vec_b : array-like, shape=[n_samples, n, n]
        base_point : array-like, shape={n_samples, n, n]

        Returns
        -------
        inner_product : float
        """
        power_euclidean = self.power_euclidean
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_a, _, _ = tangent_vec_a.shape
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
        n_tangent_vecs_b, _, _ = tangent_vec_b.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        spd_space = self.space

        assert (n_tangent_vecs_a == n_tangent_vecs_b == n_base_points
                or n_tangent_vecs_a == n_tangent_vecs_b and n_base_points == 1
                or n_base_points == n_tangent_vecs_a and n_tangent_vecs_b == 1
                or n_base_points == n_tangent_vecs_b and n_tangent_vecs_a == 1
                or n_tangent_vecs_a == 1 and n_tangent_vecs_b == 1
                or n_base_points == 1 and n_tangent_vecs_a == 1
                or n_base_points == 1 and n_tangent_vecs_b == 1)

        if n_tangent_vecs_a == 1:
            tangent_vec_a = gs.tile(
                tangent_vec_a,
                (gs.maximum(n_base_points, n_tangent_vecs_b), 1, 1))

        if n_tangent_vecs_b == 1:
            tangent_vec_b = gs.tile(
                tangent_vec_b,
                (gs.maximum(n_base_points, n_tangent_vecs_a), 1, 1))

        if n_base_points == 1:
            base_point = gs.tile(
                base_point,
                (gs.maximum(n_tangent_vecs_a, n_tangent_vecs_b), 1, 1))

        if power_euclidean == 1:
            product = gs.matmul(tangent_vec_a, tangent_vec_b)
            inner_product = gs.trace(product, axis1=1, axis2=2)
        else:
            modified_tangent_vec_a = \
                spd_space.differential_power(power_euclidean, tangent_vec_a,
                                             base_point)
            modified_tangent_vec_b = \
                spd_space.differential_power(power_euclidean, tangent_vec_b,
                                             base_point)
            product = gs.matmul(modified_tangent_vec_a, modified_tangent_vec_b)
            inner_product = gs.trace(product, axis1=1, axis2=2) \
                / (power_euclidean ** 2)

        inner_product = gs.to_ndarray(inner_product, to_ndim=2, axis=1)

        return inner_product
