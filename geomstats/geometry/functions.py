"""Module for function spaces as geometric objects."""

import math

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats._einsum import scalarvecmul
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class HilbertSphere(Manifold):
    """Class for space of one dimensional functions with norm 1.

    The tangent space is given by functions that have
    zero inner-product with the base point

    Parameters
    ----------
    domain : array-like, shape=[n_samples,]
        Grid points on the domain.

    References
    ----------
    .. [Sea2016] Srivastava, Anuj, and Eric P. Klassen.
        Functional and shape data analysis.
        Vol. 1. New York: Springer, 2016.
    """

    def __init__(self, domain, equip=True):
        self.domain = domain
        super().__init__(
            dim=math.inf,
            shape=(len(domain),),
            equip=equip,
        )

        sinf_domain = (domain - min(domain)) / (max(domain) - min(domain))
        if gs.allclose(domain, sinf_domain):
            self._sinf = self
        else:
            self._sinf = HilbertSphere(sinf_domain)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return HilbertSphereMetric

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
        return scalarvecmul(1.0 / norm, point)

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the vector space.

        This method checks the shape of the input point.

        Parameters
        ----------
        point : array-like, shape=[..., n_samples]
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

        return gs.isclose(inner_product, gs.zeros_like(inner_product), atol)

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
        return vector - scalarvecmul(coef, base_point)

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
        n_domain = len(self.domain)
        shape = (
            (n_domain,)
            if n_samples == 1
            else (
                n_samples,
                n_domain,
            )
        )
        point = gs.random.rand(*shape)

        return self.projection(point)


class HilbertSphereMetric(RiemannianMetric):
    """A Riemannian metric on the Hilbert sphere (functions of norm 1)."""

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
        x = gs.broadcast_to(self._space._sinf.domain, tangent_vec_a.shape)

        return gs.trapezoid(tangent_vec_a * tangent_vec_b, x=x, axis=-1)

    def exp(self, tangent_vec, base_point):
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
        proj_tangent_vec = self._space._sinf.to_tangent(tangent_vec, base_point)
        norm2 = self._space._sinf.metric.squared_norm(proj_tangent_vec)

        coef_1 = utils.taylor_exp_even_func(norm2, utils.cos_close_0, order=4)
        coef_2 = utils.taylor_exp_even_func(norm2, utils.sinc_close_0, order=4)
        exp = scalarvecmul(coef_1, base_point) + scalarvecmul(coef_2, proj_tangent_vec)

        return exp

    def log(self, point, base_point):
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
        squared_angle = gs.arccos(cos_angle) ** 2
        coef_1 = utils.taylor_exp_even_func(
            squared_angle, utils.inv_sinc_close_0, order=5
        )
        coef_2 = utils.taylor_exp_even_func(
            squared_angle, utils.inv_tanc_close_0, order=5
        )
        return scalarvecmul(coef_1, point) - scalarvecmul(coef_2, base_point)
