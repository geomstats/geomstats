"""Module for function spaces as geometric objects"""

import math
import pdb

import numpy as np

import geomstats.backend as gs
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class L2SpaceMetric(RiemannianMetric):
    """A Riemannian metric on the L2 space"""

    def __init__(self, domain_samples):
        self.domain = domain_samples
        self.x = (self.domain - min(self.domain)) / (
            max(self.domain) - min(self.domain)
        )
        self.dim = len(self.domain)
        super().__init__(dim=self.dim)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point."""
        inner_prod = np.trapz(tangent_vec_a * tangent_vec_b, x=self.x)
        return inner_prod

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential."""

        return base_point + tangent_vec

    def log(self, point, base_point, **kwargs):
        """Compute Riemannian logarithm of a point wrt a base point."""
        return point - base_point


class L2Space(VectorSpace):
    """Class for space of L2 functions.

    Real valued square interable functions defined on a unit interval are Hilbert spaces with a Riemannian inner product
    This class represents such manifolds.
    The L2Space (Lp in general) is a Banach Space that is a complete normed Vector space

    Ref :
    Srivastava, Anuj, and Eric P. Klassen. Functional and shape data analysis. Vol. 1. New York: Springer, 2016.
    """

    def __init__(self, domain_samples):
        self.domain = domain_samples
        self.dim = len(self.domain)
        super().__init__(shape=(self.dim,), metric=L2SpaceMetric(self.domain))


class SinfSpaceMetric(RiemannianMetric):
    """A Riemannian metric on the S_{\inf} space

    Inputs:
    -------
    domain_samples : grid points on the domain (array of shape (n_samples, ))
    """

    def __init__(self, domain_samples):
        self.domain = domain_samples
        self.x = (self.domain - min(self.domain)) / (
            max(self.domain) - min(self.domain)
        )
        self.dim = len(self.domain)
        super().__init__(dim=self.dim)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point."""

        inner_prod = np.trapz(
            tangent_vec_a * tangent_vec_b, x=self.x.reshape(1, -1), axis=1
        )

        return inner_prod

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential."""
        norm_v = self.norm(tangent_vec)
        t1 = np.cos(norm_v) * base_point
        t2 = (np.sin(norm_v) / norm_v) * base_point

        return t1 + t2

    def log(self, point, base_point, **kwargs):
        """Compute Riemannian logarithm of a point wrt a base point."""
        theta = np.arccos(self.inner_product(point, base_point, base_point))

        return (point - base_point * np.cos(theta)) * (theta / np.sin(theta))


class SinfSpace(Manifold):
    """Class for space of L2 functions with norm 1.
    The tangent space is given by functions that have zero inner-product with the base point

    Inputs:
    -------
    domain_samples : grid points on the domain (array of shape (n_samples, ))

    Ref :
    Srivastava, Anuj, and Eric P. Klassen. Functional and shape data analysis. Vol. 1. New York: Springer, 2016.
    """

    def __init__(self, domain_samples):
        self.domain = domain_samples
        super().__init__(dim=math.inf, metric=SinfSpaceMetric(self.domain))

    def projection(self, point):
        """Project a point to the infinite dimensional hypersphere.
        Parameters
        ----------
        point : array-like, shape=[..., dim]
            discrete evaluation of a function.
        Returns
        -------
        projected_point : array-like, shape=[..., dim]
            Point projected to the hypersphere.
        """

        norm_p = self.metric.norm(point)
        return point / norm_p

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the vector space.
        This method checks the shape of the input point.
        Parameters
        ----------
        point : array-like, shape=[.., {dim, [n, n]}]
            Point to test.
        atol : float

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        norms = self.metric.norm(point)

        return gs.isclose(norms - 1.0, atol)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.
        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.
        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        inner_product = self.metric.inner_product(vector, base_point)

        return gs.isclose(inner_product, atol)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.
        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        inner_product = self.metric.inner_product(vector, base_point)
        tangent_vec = vector - inner_product

        return tangent_vec

    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points on the manifold.
        If the manifold is compact, a uniform distribution is used.
        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.
        Returns
        -------
        samples : array-like, shape=[..., dim]
            Points sampled on the hypersphere.
        """
        points = gs.random.rand(n_samples, len(self.domain))

        return self.projection(points)
