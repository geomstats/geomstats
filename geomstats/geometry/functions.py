"""Module for function spaces as geometric objects"""

import math

import numpy as np

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class SinfSpaceMetric(RiemannianMetric):
    r"""A Riemannian metric on the S_{\inf} space

    Parameters:
    -------
    domain_samples : array of shape (n_samples, )
        grid points on the domain
    """

    def __init__(self, domain_samples):
        self.domain = domain_samples
        self.x = (self.domain - min(self.domain)) / (
            max(self.domain) - min(self.domain)
        )
        self.n_evals = len(self.domain)
        super().__init__(dim=self.n_evals)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., n_evals]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., n_evals]
            Tangent vector at base point.
        base_point: array-like, shape=[..., n_evals]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """

        def _f(v1, v2):
            return gs.cast(np.trapz(v1 * v2, x=self.x), gs.float32)

        inner_prod = gs.vectorize(
            (tangent_vec_a, tangent_vec_b),
            _f,
            dtype=gs.float32,
            multiple_args=True,
            signature="(i),(i)->()",
        )

        return inner_prod

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n_evals]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., n_evals]
            Point on the hypersphere.

        Returns
        -------
        exp : array-like, shape=[..., n_evals]
            Point on the hypersphere equal to the Riemannian exponential
            of tangent_vec at the base point.
        """

        def _f(v, p):
            norm_v = self.norm(v)
            t1 = gs.cos(norm_v) * p
            t2 = (gs.sin(norm_v) / norm_v) * p
            return t1 + t2

        out = gs.vectorize(
            (tangent_vec, base_point),
            _f,
            dtype=gs.float32,
            multiple_args=True,
            signature="(i),(i)->(i)",
        )

        return out

    def log(self, point, base_point, **kwargs):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., n_evals]
            Point on the hypersphere.
        base_point : array-like, shape=[..., n_evals]
            Point on the hypersphere.

        Returns
        -------
        log : array-like, shape=[..., n_evals]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """

        def _f(p0, p1):
            theta = gs.arccos(self.inner_product(p1, p0, base_point=None))
            return (p1 - p0 * gs.cos(theta)) * (theta / gs.sin(theta))

        out = gs.vectorize(
            (base_point, point),
            _f,
            dtype=gs.float32,
            multiple_args=True,
            signature="(i),(i)->(i)",
        )

        return out


class SinfSpace(Manifold):
    """Class for space of L2 functions with norm 1.

    The tangent space is given by functions that have
    zero inner-product with the base point

    Parameters:
    -------
    domain_samples : array of shape (n_samples, )
        grid points on the domain

    Ref :
    -----
    .. [Sea2016] Srivastava, Anuj, and Eric P. Klassen.
    Functional and shape data analysis.
    Vol. 1. New York: Springer, 2016.
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
