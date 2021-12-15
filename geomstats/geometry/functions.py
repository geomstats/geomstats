"""Module for function spaces as geometric objects"""

import math

import numpy as np

from geomstats.geometry.base import VectorSpace
from geomstats.geometry.euclidean import Euclidean
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

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
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
    Functional and Shape Data Analysis
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

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner-product of two tangent vectors at a base point."""
        inner_prod = np.trapz(tangent_vec_a * tangent_vec_b, x=self.x)
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


class SinfSpace(VectorSpace):
    """Class for space of L2 functions with norm 1.

    Real valued square interable functions defined on a unit interval are Hilbert spaces with a Riemannian inner product
    This class represents such manifolds.
    The L2Space (Lp in general) is a Banach Space that is a complete normed Vector space

    Ref :
    Functional and Shape Data Analysis
    """

    def __init__(self, domain_samples):
        self.domain = domain_samples
        self.dim = len(self.domain)
        super().__init__(shape=(self.dim,), metric=SinfSpaceMetric(self.domain))

    def projection(self, point):
        norm_p = self.metric.norm(point)
        return point / norm_p
