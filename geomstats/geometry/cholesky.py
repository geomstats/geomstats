"""The manifold of lower triangular matrices with positive diagonal elements"""

import math

import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry.base import OpenSet
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from geomstats.geometry.lie_group import MatrixLieGroup

class Cholesky(OpenSet):
    """Class for the manifold of lower triangular matrices with positive diagonal elements.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n, **kwargs):
        super(Cholesky, self).__init__(
            dim=int(n * (n + 1) / 2),
            metric=(n),
            ambient_space=LowerTriangularMatrices(n), **kwargs)
        self.n = n

    def belongs(self, mat, atol=gs.atol):
        """Check if a matrix is lower triangular matrix with 
        positive diagonal elements

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
        is_lower_triangular = self.ambient_space.belongs(mat, atol)
        diagonal = gs.diag(mat)
        is_positive = gs.all(diagonal > 0, axis=-1)
        belongs = gs.logical_and(is_lower_triangular, is_positive)
        return belongs

    def projection(self, point):
        """Project a matrix to the cholesksy space.

        First it is projected to space lower triangular matrices
        and then diagonal elements are exponentiated to make it positive

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.

        Returns
        -------
        projected: array-like, shape=[..., n, n]
            SPD matrix.
        """
        vec_diag = gs.exp(gs.diagonal(point, axis1 = -2, axis2 = -1))
        diag = gs.vec_to_diag(vec_diag)
        strictly_lower_triangular = Matrices.to_lower_triangular(point)
        projection = diag + strictly_lower_triangular
        return projection


class CholeskyMetric(RiemannianMetric):
    """Class for the cholesky metric on the cholesky space."""

    def __init__(self, n):
        """Build the CholeskyMetric

        Parameters
        ----------
        n : int
            Integer representing the shape of the matrices: n x n.
        

        References
        ----------
        .. [TP2019] . "Riemannian Geometry of Symmetric
        Positive Definite Matrices Via Cholesky Decomposition" 
        SIAM journal on Matrix Analysis and Applications , 2019.
         https://arxiv.org/abs/1908.09326
        """
        dim = int(n * (n + 1) / 2)
        super(CholeskyMetric, self).__init__(
            dim=dim,
            signature=(dim, 0),
            default_point_type='matrix')
        self.n = n

   

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the cholesky inner-product.

        Compute the inner-product of tangent_vec_a and tangent_vec_b
        at point base_point using the cholesky Riemannian metric.

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
        inner_product : array-like, shape=[..., n, n]
            Inner-product.
        """
        
        ip_strictly_lower = Matrices.frobenius_product
        ip_diagonal = gs.einsum()
        return ip_strictly_lower + ip_diagonal
    @staticmethod
    def _aux_exp(tangent_vec, sqrt_base_point, inv_sqrt_base_point):
        """Compute the exponential map (auxiliary function).

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
        sqrt_base_point : array-like, shape=[..., n, n]
        inv_sqrt_base_point : array-like, shape=[..., n, n]

        Returns
        -------
        exp : array-like, shape=[..., n, n]
        """
        tangent_vec_at_id = Matrices.mul(
            inv_sqrt_base_point, tangent_vec, inv_sqrt_base_point)

        tangent_vec_at_id = Matrices.to_symmetric(tangent_vec_at_id)
        exp_from_id = SymmetricMatrices.expm(tangent_vec_at_id)

        exp = Matrices.mul(
            sqrt_base_point, exp_from_id, sqrt_base_point)
        return exp

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the affine-invariant exponential map.

        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric defined in inner_product.
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
        power_affine = self.power_affine

        if power_affine == 1:
            powers = SymmetricMatrices.powerm(base_point, [1. / 2, -1. / 2])
            exp = self._aux_exp(
                tangent_vec, powers[0], powers[1])
        else:
            modified_tangent_vec = SPDMatrices.differential_power(
                power_affine, tangent_vec, base_point)
            power_sqrt_base_point = SymmetricMatrices.powerm(
                base_point, power_affine / 2)
            power_inv_sqrt_base_point = GeneralLinear.inverse(
                power_sqrt_base_point)
            exp = self._aux_exp(
                modified_tangent_vec,
                power_sqrt_base_point,
                power_inv_sqrt_base_point)
            exp = SymmetricMatrices.powerm(exp, 1 / power_affine)

        return exp

    @staticmethod
    def _aux_log(point, sqrt_base_point, inv_sqrt_base_point):
        """Compute the log (auxiliary function).

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
        sqrt_base_point : array-like, shape=[..., n, n]
        inv_sqrt_base_point : array-like, shape=[.., n, n]

        Returns
        -------
        log : array-like, shape=[..., n, n]
        """
        point_near_id = Matrices.mul(
            inv_sqrt_base_point, point, inv_sqrt_base_point)
        point_near_id = Matrices.to_symmetric(point_near_id)

        log_at_id = SPDMatrices.logm(point_near_id)
        log = Matrices.mul(sqrt_base_point, log_at_id, sqrt_base_point)
        return log

    def log(self, point, base_point, **kwargs):
        """Compute the affine-invariant logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the metric defined in inner_product.
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
            Riemannian logarithm of point at base_point.
        """
        power_affine = self.power_affine

        if power_affine == 1:
            powers = SymmetricMatrices.powerm(base_point, [1. / 2, -1. / 2])
            log = self._aux_log(point, powers[0], powers[1])
        else:
            power_point = SymmetricMatrices.powerm(point, power_affine)
            powers = SymmetricMatrices.powerm(
                base_point, [power_affine / 2, -power_affine / 2])
            log = self._aux_log(
                power_point, powers[0], powers[1])
            log = SPDMatrices.inverse_differential_power(
                power_affine, log, base_point)
        return log

    def parallel_transport(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector a
        along the geodesic defined by exp_(base_point)(tangent_vec_b).
        Denoting `tangent_vec_a` by `S`, `base_point` by `A`, let
        `B = Exp_A(tangent_vec_b)` and :math: `E = (BA^{- 1})^({ 1 / 2})`.
        Then the
        parallel transport to `B`is:

        ..math::
                        S' = ESE^T

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            Tangent vector at base point to be transported.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Tangent vector at base point, initial speed of the geodesic along
            which the parallel transport is computed.
        base_point : array-like, shape=[..., dim + 1]
            Point on the manifold of SPD matrices.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., dim + 1]
            Transported tangent vector at exp_(base_point)(tangent_vec_b).
        """
        end_point = self.exp(tangent_vec_b, base_point)
        inverse_base_point = GeneralLinear.inverse(base_point)
        congruence_mat = Matrices.mul(end_point, inverse_base_point)
        congruence_mat = gs.linalg.sqrtm(congruence_mat)
        return Matrices.congruent(tangent_vec_a, congruence_mat)

