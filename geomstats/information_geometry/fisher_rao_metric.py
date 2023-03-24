"""Class to implement simply the Fisher-Rao metric on information manifolds."""

from scipy.integrate import quad_vec

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric


class FisherRaoMetric(RiemannianMetric):
    r"""Class to derive the information metric from the pdf in InformationManifoldMixin.

    Given a statistical manifold with coordinates :math:`\theta`,
    the Fisher information metric is:
    :math:`g_{j k}(\theta)=\int_X \frac{\partial \log p(x, \theta)}
        {\partial \theta_j}\frac{\partial \log p(x, \theta)}
        {\partial \theta_k} p(x, \theta) d x`

    Attributes
    ----------
    space : InformationManifold
        Riemannian Manifold for a parametric family of (real) distributions.
    support : list, shape = (2,)
        Left and right bounds for the support of the distribution.
        But this is just to help integration, bounds should be as large as needed.
    """

    def __init__(self, space, support):
        super().__init__(
            space=space,
            signature=(space.dim, 0),
        )
        self.support = support

    def metric_matrix(self, base_point):
        r"""Compute the inner-product matrix.

        The Fisher information matrix is noted I in the literature.

        Compute the inner-product matrix of the Fisher information metric
        at the tangent space at base point.

        .. math::

            I(\theta) = \mathbb{E}[YY^T]

        where,

        ..math::

            Y = \nabla_{\theta} \log f_{\theta}(x)


        After manipulation and in indicial notation

        .. math::
             I_{ij} = \int \
             \partial_{i} f_{\theta}(x)\
             \partial_{j} f_{\theta}(x)\
             \frac{1}{f_{\theta}(x)}


        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        metric_mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.

        References
        ----------
        .. [AS1985] Amari, S (1985)
            Differential Geometric Methods in Statistics, Berlin, Springer – Verlag.
        """

        def pdf(x):
            """Compute pdf at a fixed point on the support.

            Parameters
            ----------
            x : float, shape (,)
                Point on the support of the distribution
            """
            return lambda point: gs.squeeze(self._space.point_to_pdf(point)(x), axis=-1)

        def _function_to_integrate(x):
            pdf_x = pdf(x)
            (
                pdf_x_at_base_point,
                pdf_x_derivative_at_base_point,
            ) = gs.autodiff.value_and_grad(pdf_x, to_numpy=True)(base_point)

            return gs.einsum(
                "...ij,...->...ij",
                gs.einsum(
                    "...i,...j->...ij",
                    pdf_x_derivative_at_base_point,
                    pdf_x_derivative_at_base_point,
                ),
                1 / pdf_x_at_base_point,
            )

        return quad_vec(_function_to_integrate, *self.support)[0]

    def inner_product_derivative_matrix(self, base_point):
        r"""Compute the derivative of the inner-product matrix.

        Compute the derivative of the inner-product matrix of
        the Fisher information metric at the tangent space at base point.

        .. math::

            \partial_{\theta} I(\theta) = \partial_{\theta} \mathbb{E}[YY^T]

        where,

        ..math::

            Y = \nabla_{\theta} \log f_{\theta}(x)

        or, in indicial notation:

        .. math::

            \partial_k I_{ij} = \int\
            \partial_{ki}^2 f\partial_j f \frac{1}{f} + \
            \partial_{kj}^2 f\partial_i f \frac{1}{f} - \
            \partial_i f \partial_j f \partial_k f \frac{1}{f^2}

        with :math:`f = f_{\theta}(x)`


        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim, dim]
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: math:`mat_{ijk} = \partial_k g_{ij}`.
        """

        def pdf(x):
            """Compute pdf at a fixed point on the support.

            Parameters
            ----------
            x : float, shape (,)
                Point on the support of the distribution
            """
            return lambda point: gs.squeeze(self._space.point_to_pdf(point)(x), axis=-1)

        def _function_to_integrate(x):
            pdf_x = pdf(x)
            (
                pdf_x_at_base_point,
                pdf_x_derivative_at_base_point,
                pdf_x_hessian_at_base_point,
            ) = gs.autodiff.value_jacobian_and_hessian(pdf_x)(base_point)

            return gs.einsum(
                "...,...ijk->...ijk",
                1 / (pdf_x_at_base_point**2),
                gs.einsum(
                    "...,...ijk->...ijk",
                    pdf_x_at_base_point,
                    gs.einsum(
                        "...ki,...j->...ijk",
                        pdf_x_hessian_at_base_point,
                        pdf_x_derivative_at_base_point,
                    )
                    + gs.einsum(
                        "...kj,...i->...ijk",
                        pdf_x_hessian_at_base_point,
                        pdf_x_derivative_at_base_point,
                    ),
                )
                - gs.einsum(
                    "...i, ...j, ...k -> ...ijk",
                    pdf_x_derivative_at_base_point,
                    pdf_x_derivative_at_base_point,
                    pdf_x_derivative_at_base_point,
                ),
            )

        return quad_vec(_function_to_integrate, *self.support)[0]
