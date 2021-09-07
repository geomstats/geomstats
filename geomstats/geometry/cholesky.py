"""The manifold of lower triangular matrices with positive diagonal elements"""

import math

import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry.base import OpenSet
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices

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

   