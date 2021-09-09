"""The manifold of lower triangular matrices with positive diagonal elements"""

import math

import geomstats.backend as gs
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
            metric=(CholeskyMetric),
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

   
    def diag_inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner product using only diagonal elements.

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
        ip_diagonal : array-like, shape=[...]
            Inner-product.
        """
        
        inv_sqrt_diagonal = gs.power(gs.diagonal(base_point), -2)
        ip_diagonal = gs.einsum("...ii,...ii ,...i->...", 
            tangent_vec_a, tangent_vec_b, inv_sqrt_diagonal)
        return ip_diagonal

    def strictly_lower_inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner product using only strictly lower triangular elements.

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
        ip_sl : array-like, shape=[...]
            Inner-product.
        """
        
        sl_tagnet_vec_a = gs.tril_to_vec(tangent_vec_a)
        sl_tagnet_vec_b = gs.tril_to_vec(tangent_vec_b)
        ip_sl = gs.einsum(
            "...i, ...i-> ....", sl_tagnet_vec_a, sl_tagnet_vec_b)
        return ip_sl


    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner product using only strictly lower triangular elements.

        
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
        inner_product : array-like, shape=[...]
            Inner-product.
        """
        diag_inner_product = self.diag_inner_product(
            tangent_vec_a, tangent_vec_b, base_point) 
        strictly_lower_inner_product = self.strictly_lower_inner_product(
            tangent_vec_a, tangent_vec_b, base_point)
        return diag_inner_product + strictly_lower_inner_product

    
    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Cholesky exponential map.

        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the Cholesky metric.
        This gives a lower triangular matrix with positive elements.

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
        sl_base_point = Matrices.to_strictly_lower_triangular(base_point)
        sl_tangent_vec = Matrices.to_strictly_lower_triangular(tangent_vec)
        diag_base_point = Matrices.to_diagonal(base_point)
        diag_tangent_vec = Matrices.to_diagonal(tangent_vec)

        sl_exp = 
        diag_exp =
        exp = sl_exp + diag_exp
        return exp

    def log(self, point, base_point, **kwargs):
        """Compute the Cholesky logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the Cholesky metric.
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
        sl_base_point = Matrices.to_strictly_lower_triangular(base_point)
        sl_tangent_vec = Matrices.to_strictly_lower_triangular(tangent_vec)
        diag_base_point = Matrices.to_diagonal(base_point)
        diag_tangent_vec = Matrices.to_diagonal(tangent_vec)

        sl_log = 
        diag_log =
        log = sl_log + diag_log
        return log

     