"""The manifold of Positive Semi Definite matrices of rank k."""

import math

import geomstats._backend as gs
import geomstats.vectorization
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


class PSDMatricesRankK(Manifold):
    """Class for the manifold of symmetric positive definite (PSD) matrices.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    k: rank
        Integer representing the rank of the matrix (k<n)
    """

    def __init__(
        self,
        n,
        k,
        metric=None,
        default_point_type="matrix",
        default_coords_type="intrinsic",
        **kwargs
    ):
        # ANNA how to check for this?
        if k == n:
            print(
                "Initialize a Symmetric Positive Definite Matrix as the rank is equal to the dimension"
            )
        super(Manifold, self).__init__(**kwargs)
        self.n = n
        self.dim = (int(n * (n + 1) / 2),)
        self.default_point_type = default_point_type
        self.default_coords_type = default_coords_type
        self.metric = metric
        self.rank = k
        self.sym = SymmetricMatrices(self.n)

    def belongs(self, mat, atol=gs.atol):
        """Check if a matrix is symmetric with positive eigenvalues and
         with n-k zero eigenvalues.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if mat is an SPD matrix.
        """
        is_symmetric = self.sym.belongs(mat, atol)
        eigvalues = gs.linalg.eigvalsh(mat)
        is_semipositive = gs.all(eigvalues > -gs.atol, axis=-1)
        is_rankk = gs.linalg.matrix_rank(mat) == self.rank
        belongs = gs.logical_and(
            gs.logical_and(is_symmetric, is_semipositive), is_rankk
        )
        return belongs

    def projection(self, point):
        """Project a matrix to the space of PSD matrices of rank k.

        First the symmetric part of point is computed, then the eigenvalues
        are floored to zeros. To ensure rank k, n-k eigenvalues are set to 0

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.

        Returns
        -------
        projected: array-like, shape=[..., n, n]
            PSD matrix.
        """

        sym = Matrices(self.n, self.n).to_symmetric(point)
        eigvals, eigvecs = gs.linalg.eigh(sym)
        regularized = gs.where(eigvals < 0, 0, eigvals)
        regularized[0 : (self.n - self.rank)] = [0] * (self.n - self.rank)
        reconstruction = gs.einsum("...ij,...j->...ij", eigvecs, regularized)
        return Matrices.mul(reconstruction, Matrices.transpose(eigvecs))
        # ANNA - how can we handle this case?
        # the rank is lower because there are more than n-k zeros eigenvalues

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in PSD(n,k) from the log-uniform distribution of SPD matrices
        and adding zero eigenvalues.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample in the tangent space.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled in PSD(n,k).
        """
        n = self.n
        size = (n_samples, n, n) if n_samples != 1 else (n, n)
        mat = bound * (2 * gs.random.rand(*size) - 1)
        spd_mat = GeneralLinear.exp(Matrices.to_symmetric(mat))
        if n_samples > 1:
            psd_mat = [self.projection(i) for i in spd_mat]
        else:
            psd_mat = [self.projection(spd_mat)]
        return psd_mat

    # ANNA add the correct citation of Yann's work

    def is_tangent(self, vector, base_point):
        """Check if the vector belongs to the tangent space at the input point.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix to check if it belongs to the tangent space.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.
            Optional, default: None.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if vector belongs to tangent space at base_point.
        References
        ----------
        .. [TP2019] Thanwerdas, Pennec. "Is affine-invariance well defined on
          SPD matrices? A principled continuum of metrics" Proc. of GSI, 2019.
          https://arxiv.org/abs/1906.01349
        """

        vector_sym = [
            vector if self.sym.belongs(vector) else self.sym.projection(vector)
        ][0]
        # check if symmetric
        r, delta, rt = gs.linalg.svd(base_point)
        rort = r[:, self.n - self.rank : self.n]
        rort_t = rt[self.n - self.rank : self.n, :]
        check = gs.matmul(
            gs.matmul(gs.matmul(rort, rort_t), vector_sym), gs.matmul(rort, rort_t)
        )
        if (
            gs.logical_and(
                gs.less_equal(check, -gs.atol), gs.greater(check, gs.atol)
            ).sum()
            == 0
        ):
            return True
        else:
            return False

    def to_tangent(self, vector, base_point):
        """Project the input vector to the tangent space of PSD(n,k) at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix to check if it belongs to the tangent space.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.
            Optional, default: None.

        Returns
        -------
        tangent : array-like, shape=[...,n,n]
            Projection of the tangent vector at base_point.
        """
        if self.is_tangent(vector, base_point):
            return vector
        else:
            vector_sym = [
                vector if self.sym.belongs(vector) else self.sym.projection(vector)
            ][0]
            r, delta, rt = gs.linalg.svd(base_point)
            rort = r[:, self.n - self.rank : self.n]
            rort_t = rt[self.n - self.rank : self.n, :]
            return (
                gs.matmul(
                    gs.matmul(gs.matmul(rort, rort_t), vector_sym),
                    gs.matmul(rort, rort_t),
                )
                + vector_sym
            )


class PSDMetricBuresWasserstein(RiemannianMetric):
    """Class for the Bures-Wasserstein metric on the PSD manifold.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.

    References
    ----------
    .. [BJL2017]_ Bhatia, Jain, Lim. "On the Bures-Wasserstein distance between
      positive definite matrices" Elsevier, Expositiones Mathematicae,
      vol. 37(2), 165-191, 2017. https://arxiv.org/pdf/1712.01504.pdf
    .. [MMP2018]_ Malago, Montrucchio, Pistone. "Wasserstein-Riemannian
      geometry of Gaussian densities"  Information Geometry, vol. 1, 137-179,
      2018. https://arxiv.org/pdf/1801.09269.pdf
    """

    def __init__(self, n):
        dim = int(n * (n + 1) / 2)
        super(PSDMetricBuresWasserstein, self).__init__(
            dim=dim, signature=(dim, 0), default_point_type="matrix"
        )
        self.n = n

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the Bures-Wasserstein inner-product.

        Compute the inner-product of tangent_vec_a :math: `A` and tangent_vec_b
        :math: `B` at point base_point :math: `S=PDP^\top` using the
        Bures-Wasserstein Riemannian metric:
        ..math::
        `\frac{1}{2}\sum_{i,j}\frac{[P^\top AP]_{ij}[P^\top BP]_{ij}}{d_i+d_j}`
        .

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        eigvals, eigvecs = gs.linalg.eigh(base_point)
        transp_eigvecs = Matrices.transpose(eigvecs)
        rotated_tangent_vec_a = Matrices.mul(transp_eigvecs, tangent_vec_a, eigvecs)
        rotated_tangent_vec_b = Matrices.mul(transp_eigvecs, tangent_vec_b, eigvecs)

        coefficients = 1 / (eigvals[..., :, None] + eigvals[..., None, :])
        result = (
            Matrices.frobenius_product(
                coefficients * rotated_tangent_vec_a, rotated_tangent_vec_b
            )
            / 2
        )

        return result

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Bures-Wasserstein exponential map.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp : array-like, shape=[...,]
            Riemannian exponential.
        """
        eigvals, eigvecs = gs.linalg.eigh(base_point)
        transp_eigvecs = Matrices.transpose(eigvecs)
        rotated_tangent_vec = Matrices.mul(transp_eigvecs, tangent_vec, eigvecs)
        coefficients = 1 / (eigvals[..., :, None] + eigvals[..., None, :])
        rotated_sylvester = rotated_tangent_vec * coefficients
        rotated_hessian = gs.einsum("...ij,...j->...ij", rotated_sylvester, eigvals)
        rotated_hessian = Matrices.mul(rotated_hessian, rotated_sylvester)
        hessian = Matrices.mul(eigvecs, rotated_hessian, transp_eigvecs)

        return base_point + tangent_vec + hessian

    def log(self, point, base_point, **kwargs):
        """Compute the Bures-Wasserstein logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the Bures-Wasserstein metric.
        This gives a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., n, n]
            Riemannian logarithm.
        """
        product = gs.matmul(base_point, point)
        sqrt_product = gs.linalg.sqrtm(product)
        transp_sqrt_product = Matrices.transpose(sqrt_product)

        return sqrt_product + transp_sqrt_product - 2 * base_point

    def squared_dist(self, point_a, point_b, **kwargs):
        """Compute the Bures-Wasserstein squared distance.

        Compute the Riemannian squared distance between point_a and point_b.

        Parameters
        ----------
        point_a : array-like, shape=[..., n, n]
            Point.
        point_b : array-like, shape=[..., n, n]
            Point.

        Returns
        -------
        squared_dist : array-like, shape=[...]
            Riemannian squared distance.
        """
        product = gs.matmul(point_a, point_b)
        sqrt_product = gs.linalg.sqrtm(product)
        trace_a = gs.trace(point_a, axis1=-2, axis2=-1)
        trace_b = gs.trace(point_b, axis1=-2, axis2=-1)
        trace_prod = gs.trace(sqrt_product, axis1=-2, axis2=-1)

        return trace_a + trace_b - 2 * trace_prod


class PSDMetricEuclidean(RiemannianMetric):
    """Class for the Euclidean metric on the SPD manifold."""

    def __init__(self, n, power_euclidean=1):
        dim = int(n * (n + 1) / 2)
        super(SPDMetricEuclidean, self).__init__(
            dim=dim, signature=(dim, 0), default_point_type="matrix"
        )
        self.n = n
        self.power_euclidean = power_euclidean

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the Euclidean inner-product.

        Compute the inner-product of tangent_vec_a and tangent_vec_b
        at point base_point using the power-Euclidean metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        power_euclidean = self.power_euclidean
        spd_space = SPDMatrices

        if power_euclidean == 1:
            inner_product = Matrices.frobenius_product(tangent_vec_a, tangent_vec_b)
        else:
            modified_tangent_vec_a = spd_space.differential_power(
                power_euclidean, tangent_vec_a, base_point
            )
            modified_tangent_vec_b = spd_space.differential_power(
                power_euclidean, tangent_vec_b, base_point
            )

            inner_product = Matrices.frobenius_product(
                modified_tangent_vec_a, modified_tangent_vec_b
            ) / (power_euclidean ** 2)
        return inner_product

    @staticmethod
    @geomstats.vectorization.decorator(["matrix", "matrix"])
    def exp_domain(tangent_vec, base_point):
        """Compute the domain of the Euclidean exponential map.

        Compute the real interval of time where the Euclidean geodesic starting
        at point `base_point` in direction `tangent_vec` is defined.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp_domain : array-like, shape=[..., 2]
            Interval of time where the geodesic is defined.
        """
        invsqrt_base_point = SymmetricMatrices.powerm(base_point, -0.5)

        reduced_vec = gs.matmul(invsqrt_base_point, tangent_vec)
        reduced_vec = gs.matmul(reduced_vec, invsqrt_base_point)
        eigvals = gs.linalg.eigvalsh(reduced_vec)
        min_eig = gs.amin(eigvals, axis=1)
        max_eig = gs.amax(eigvals, axis=1)
        inf_value = gs.where(max_eig <= 0.0, gs.array(-math.inf), -1.0 / max_eig)
        inf_value = gs.to_ndarray(inf_value, to_ndim=2)
        sup_value = gs.where(min_eig >= 0.0, gs.array(-math.inf), -1.0 / min_eig)
        sup_value = gs.to_ndarray(sup_value, to_ndim=2)
        domain = gs.concatenate((inf_value, sup_value), axis=1)

        return domain


class PSDMetricLogEuclidean(RiemannianMetric):
    """Class for the Log-Euclidean metric on the SPD manifold.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        dim = int(n * (n + 1) / 2)
        super(SPDMetricLogEuclidean, self).__init__(
            dim=dim, signature=(dim, 0), default_point_type="matrix"
        )
        self.n = n

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the Log-Euclidean inner-product.

        Compute the inner-product of tangent_vec_a and tangent_vec_b
        at point base_point using the log-Euclidean metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        spd_space = SPDMatrices

        modified_tangent_vec_a = spd_space.differential_log(tangent_vec_a, base_point)
        modified_tangent_vec_b = spd_space.differential_log(tangent_vec_b, base_point)
        product = Matrices.trace_product(modified_tangent_vec_a, modified_tangent_vec_b)
        return product

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Log-Euclidean exponential map.

        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the Log-Euclidean metric.
        This gives a symmetric positive definite matrix.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., n, n]
            Riemannian exponential.
        """
        log_base_point = SPDMatrices.logm(base_point)
        dlog_tangent_vec = SPDMatrices.differential_log(tangent_vec, base_point)
        exp = SymmetricMatrices.expm(log_base_point + dlog_tangent_vec)

        return exp

    def log(self, point, base_point, **kwargs):
        """Compute the Log-Euclidean logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the Log-Euclidean metric.
        This gives a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., n, n]
            Riemannian logarithm.
        """
        log_base_point = SPDMatrices.logm(base_point)
        log_point = SPDMatrices.logm(point)
        log = SPDMatrices.differential_exp(log_point - log_base_point, log_base_point)

        return log
