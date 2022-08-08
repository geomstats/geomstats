"""Class to implement simply the Fisher-Rao metric on information manifolds."""

from scipy.integrate import quad_vec

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric


class FisherRaoMetric(RiemannianMetric):
    """Class to derive the information metric from the pdf in InformationManifold."""

    def __init__(self, information_manifold, support):
        self.information_manifold = information_manifold
        self.support = support

    def metric_matrix(self, base_point):
        r"""Compute the inner-product matrix.

        The Fisher information matrix is noted I in the literature.

        Compute the inner-product matrix of the Fisher information metric
        at the tangent space at base point.

        :math: `I(\theta) = \mathbb{E}[(\partial_{\theta}(\log(pdf_{\theta}(x))))^2]`

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.

        References
        ----------
        .. [AS1985] Amari, S (1985)
            Differential Geometric Methods in Statistics, Berlin, Springer – Verlag.
        """

        def log_pdf_at_x(x):
            return lambda point: gs.log(
                self.information_manifold.point_to_pdf(point)(x)
            )

        def sqrtscore(x):
            return gs.autodiff.jacobian(log_pdf_at_x(x))(base_point)

        def score(x):
            return gs.einsum("...i, ...j -> ...ij", sqrtscore(x), sqrtscore(x))

        return quad_vec(
            lambda x: score(x) * self.information_manifold.point_to_pdf(base_point)(x),
            *self.support
        )[0]

    def inner_product_derivative_matrix(self, base_point):
        r"""Compute the derivative of the inner-product matrix.

        Compute the derivative of the inner-product matrix of
        the Fisher information metric at the tangent space at base point.

        :math: `I(\theta) =
         \partial_{\theta} \mathbb{E}[(\partial_{\theta}(\log(f_{\theta}(x))))^2]`

        With further calculations, this is:
        :math: `int [\frac{2 *
         \partial_{\theta}^2 f \times \partial_{\theta} f \times f +
         (\partial_{\theta} f)^3}{f^2}]

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Derivative of the inner-product matrix.
        """

        def pdf(x):
            return lambda point: self.information_manifold.point_to_pdf(point)(x)

        function = (
            lambda x: 1
            / (pdf(x)(base_point) ** 2)
            * (
                2
                * pdf(x)(base_point)
                * gs.einsum(
                    "...ij, ...k -> ...ijk",
                    gs.autodiff.jacobian(gs.autodiff.jacobian(pdf(x)))(base_point),
                    gs.autodiff.jacobian(pdf(x))(base_point),
                )
                + gs.einsum(
                    "...i, ...j, ...k -> ...ijk",
                    gs.autodiff.jacobian(pdf(x))(base_point),
                    gs.autodiff.jacobian(pdf(x))(base_point),
                    gs.autodiff.jacobian(pdf(x))(base_point),
                )
            )
        )

        return quad_vec(function, *self.support)[0]
