"""Variants of the metric on the Hilbert sphere."""

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats._einsum import scalarvecmul
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import repeat_out


class TrapezoidHilbertSphereMetric(RiemannianMetric):
    """Metric on the Hilbert sphere with closed forms.

    Variant of :class:`geomstats.geometry.functions.HilbertSphereMetric`
    that computes the inner product with the trapezoidal rule, on the domain
    rescaled to :math:`[0, 1]`, and the exponential, the logarithm, the
    parallel transport and the injectivity radius with the closed forms of the
    sphere, instead of pulling back the metric of the unit sphere.

    Parameters
    ----------
    space : HilbertSphere
        Manifold.
    """

    def __init__(self, space):
        super().__init__(space)
        domain = space.domain
        self.rescaled_domain = (domain - min(domain)) / (max(domain) - min(domain))

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
        x = gs.broadcast_to(self.rescaled_domain, tangent_vec_a.shape)

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
        coef = self.inner_product(tangent_vec, base_point) / self.squared_norm(
            base_point
        )
        proj_tangent_vec = tangent_vec - scalarvecmul(coef, base_point)
        norm2 = self.squared_norm(proj_tangent_vec)

        coef_1 = utils.taylor_exp_even_func(norm2, utils.cos_close_0, order=4)
        coef_2 = utils.taylor_exp_even_func(norm2, utils.sinc_close_0, order=4)
        return scalarvecmul(coef_1, base_point) + scalarvecmul(coef_2, proj_tangent_vec)

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

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        The geodesic from :math:`x` with direction :math:`v` stays in the
        plane spanned by :math:`x` and :math:`u = v / \|v\|`, in which it
        rotates by the angle :math:`\theta = \|v\|`. The component of the
        tangent vector :math:`w` along :math:`u` rotates with it, and the
        orthogonal one is unchanged:

        .. math::

            P(w) = w - \langle w, u \rangle
            (\sin(\theta) x + (1 - \cos(\theta)) u)

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n_samples]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., n_samples]
            Point on the hypersphere.
        direction : array-like, shape=[..., n_samples]
            Tangent vector at base point, along which the parallel transport
            is computed.
        end_point : array-like, shape=[..., n_samples]
            Point on the hypersphere to transport to. Unused if ``direction``
            is given.

        Returns
        -------
        transported_tangent_vec : array-like, shape=[..., n_samples]
            Transported tangent vector at ``exp_(base_point)(direction)``.
        """
        if direction is None:
            if end_point is None:
                raise ValueError(
                    "Either an end_point or a direction must be given to define the"
                    " geodesic along which to transport."
                )
            direction = self.log(end_point, base_point)

        theta = self.norm(direction)
        safe_theta = gs.where(theta == 0.0, 1.0, theta)
        unit_direction = scalarvecmul(1.0 / safe_theta, direction)
        coef = self.inner_product(tangent_vec, unit_direction)
        return (
            tangent_vec
            - scalarvecmul(coef * gs.sin(theta), base_point)
            - scalarvecmul(coef * (1.0 - gs.cos(theta)), unit_direction)
        )

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        Geodesics are great circles, so that the cut point of a point is its
        antipode, and the radius is pi everywhere.

        Parameters
        ----------
        base_point : array-like, shape=[..., n_samples]
            Point on the hypersphere.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(gs.pi)
        return repeat_out(self._space.point_ndim, radius, base_point)
