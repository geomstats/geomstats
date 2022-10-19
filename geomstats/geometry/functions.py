"""Module for function spaces as geometric objects."""

import math

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class HilbertSphereMetric(RiemannianMetric):
    r"""A Riemannian metric on the Hilbert sphere (functions of norm 1).

    Parameters
    ----------
    domain_samples : array of shape (n_samples, )
        Grid points on the domain.
    """

    def __init__(self, domain_samples):
        self.domain = domain_samples
        self.x = (self.domain - min(self.domain)) / (
            max(self.domain) - min(self.domain)
        )
        self.n_evals = len(self.domain)
        super().__init__(dim=self.n_evals)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n_samples]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n_samples]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., n_samples], optional
            Point on the hypersphere.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        tangent_vec_a, tangent_vec_b = gs.broadcast_arrays(tangent_vec_a, tangent_vec_b)
        x = gs.broadcast_to(self.x, tangent_vec_a.shape)

        l2_norm = gs.trapz(tangent_vec_a * tangent_vec_b, x=x, axis=-1)

        return l2_norm

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n_samples]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., n_samples]
            Point on the hypersphere.

        Returns
        -------
        exp : array-like, shape=[..., n_samples]
            Point on the hypersphere equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        sinf = HilbertSphere(self.x)
        proj_tangent_vec = sinf.to_tangent(tangent_vec, base_point)
        norm = sinf.metric.norm(proj_tangent_vec)
        norm = gs.clip(norm, -gs.pi, gs.pi)
        coef_1 = utils.taylor_exp_even_func(norm, utils.cos_close_0, order=4)
        coef_2 = utils.taylor_exp_even_func(norm, utils.sinc_close_0, order=4)
        exp = gs.einsum("...,...j->...j", coef_1, base_point) + gs.einsum(
            "...,...j->...j", coef_2, proj_tangent_vec
        )

        return exp

    def log(self, point, base_point, **kwargs):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., n_samples]
            Point on the hypersphere.
        base_point : array-like, shape=[..., n_samples]
            Point on the hypersphere.

        Returns
        -------
        log : array-like, shape=[..., n_samples]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        inner_prod = self.inner_product(base_point, point)
        cos_angle = gs.clip(inner_prod, -1.0, 1.0)
        theta = gs.arccos(cos_angle)
        coef_1_ = utils.taylor_exp_even_func(theta, utils.inv_sinc_close_0, order=5)
        coef_2_ = utils.taylor_exp_even_func(theta, utils.inv_tanc_close_0, order=5)
        log = gs.einsum("...,...j->...j", theta * coef_1_, point) - gs.einsum(
            "...,...j->...j", theta * coef_2_, base_point
        )

        return log


class HilbertSphere(Manifold):
    """Class for space of one dimensional functions with norm 1.

    The tangent space is given by functions that have
    zero inner-product with the base point

    Parameters
    ----------
    domain_samples : array of shape (n_samples, )
        Grid points on the domain.

    Ref :
    -----
    .. [Sea2016] Srivastava, Anuj, and Eric P. Klassen.
    Functional and shape data analysis.
    Vol. 1. New York: Springer, 2016.
    """

    def __init__(self, domain_samples):
        self.domain = domain_samples
        super().__init__(
            dim=math.inf,
            shape=(1, len(domain_samples)),
            metric=HilbertSphereMetric(self.domain),
        )

    def projection(self, point):
        """Project a point to the infinite dimensional hypersphere.

        Parameters
        ----------
        point : array-like, shape=[..., n_samples]
            Discrete evaluation of a function.

        Returns
        -------
        projected_point : array-like, shape=[..., n_samples]
            Point projected to the hypersphere.
        """
        norm = self.metric.norm(point)
        projected_point = gs.einsum("...,...i->...i", 1.0 / norm, point)

        return projected_point

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the vector space.

        This method checks the shape of the input point.

        Parameters
        ----------
        point : array-like, shape=[.., n_samples]
            Point to test.
        atol : float

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        norms = self.metric.norm(point)

        return gs.isclose(norms, 1.0, atol)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., n_samples]
            Vector.
        base_point : array-like, shape=[..., n_samples]
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
        vector : array-like, shape=[..., n_samples]
            Vector.
        base_point : array-like, shape=[..., n_samples]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n_samples]
            Tangent vector at base point.
        """
        sq_norm = self.metric.squared_norm(base_point)
        inner_product = self.metric.inner_product(vector, base_point)
        coef = inner_product / sq_norm
        tangent_vec = vector - gs.einsum("...,...j->...j", coef, base_point)

        return tangent_vec

    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points on the manifold.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            This parameter is ignored

        Returns
        -------
        samples : array-like, shape=[..., dim]
        """
        points = gs.random.rand(n_samples, len(self.domain))

        return self.projection(points)
