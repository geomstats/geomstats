"""Statistical Manifold of Dirichlet distributions with the Fisher metric.

Lead author: Alice Le Brigant.
"""

import math

from scipy.stats import dirichlet

import geomstats.backend as gs
from geomstats._einsum import scalarmatmul
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.positive_orthant import PositiveOrthant
from geomstats.geometry.riemannian_metric import CoordinateRiemannianMetric
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyMultivariateRandomVariable,
)
from geomstats.numerics.bvp import ScipySolveBVP
from geomstats.numerics.geodesic import ExpODESolver, LogODESolver
from geomstats.numerics.ivp import ScipySolveIVP
from geomstats.vectorization import repeat_out


class DirichletDistributions(InformationManifoldMixin, PositiveOrthant):
    r"""Class for the manifold of Dirichlet distributions.

    This is Dirichlet = :math:`(R_+^*)^dim`, the positive quadrant of the
    dim-dimensional Euclidean space.

    Parameters
    ----------
    dim : int
        Dimension of the manifold of Dirichlet distributions.
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, dim, equip=True):
        super().__init__(dim=dim, support_shape=(dim,), equip=equip)
        self._scp_rv = DirichletRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return DirichletMetric

    def sample(self, point, n_samples=1):
        """Sample from the Dirichlet distribution.

        Sample from the Dirichlet distribution with parameters provided
        by point. This gives n_samples points in the simplex.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a Dirichlet distribution.
        n_samples : int
            Number of points to sample for each set of parameters in point.

        Returns
        -------
        samples : array-like, shape=[..., n_samples, dim]
            Sample from the Dirichlet distributions.
        """
        return self._scp_rv.rvs(point, n_samples)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the Dirichlet
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a beta distribution.

        Returns
        -------
        pdf : function
            Probability density function of the Dirichlet distribution with
            parameters provided by point.
        """
        return lambda x: self._scp_rv.pdf(x, point=point)


class DirichletMetric(CoordinateRiemannianMetric):
    """Class for the Fisher information metric on Dirichlet distributions."""

    def __init__(self, space):
        super().__init__(space=space)

        self.log_solver = LogODESolver(
            space, n_nodes=10, integrator=ScipySolveBVP(max_nodes=1000)
        )
        self.exp_solver = ExpODESolver(space, integrator=ScipySolveIVP(method="LSODA"))

    def metric_matrix(self, base_point):
        """Compute the inner-product matrix.

        Compute the inner-product matrix of the Fisher information metric
        at the tangent space at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        batch_shape = base_point.shape[:-1]

        mat_ones = gs.ones(batch_shape + (self._space.dim, self._space.dim))
        poly_sum = gs.polygamma(1, gs.sum(base_point, axis=-1))
        mat_diag = from_vector_to_diagonal_matrix(gs.polygamma(1, base_point))

        return mat_diag - scalarmatmul(poly_sum, mat_ones)

    def inner_product_derivative_matrix(self, base_point):
        r"""Compute derivative of the inner prod matrix at base point.

        .. math::

            \partial_k g_{ij} = \psi''(\alpha_i) \delta_{ij} \delta_{ik}
            - \psi''\left(\sum_l \alpha_l\right)

        where :math:`\psi` is the digamma function.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        metric_derivative : array-like, shape=[..., dim, dim, dim]
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: :math:`mat_{ijk} = \partial_k g_{ij}`.
        """
        dim = self._space.dim
        eye = gs.eye(dim, dtype=base_point.dtype)
        diagonal_part = gs.einsum(
            "...i,ij,ik->...ijk", gs.polygamma(2, base_point), eye, eye
        )
        sum_part = gs.einsum(
            "...,ijk->...ijk",
            gs.polygamma(2, gs.sum(base_point, axis=-1)),
            gs.ones((dim, dim, dim), dtype=base_point.dtype),
        )
        return diagonal_part - sum_part

    def christoffels(self, base_point):
        r"""Compute the Christoffel symbols.

        Compute the Christoffel symbols of the Fisher information metric:

        .. math::

            \Gamma_{j k}^i=\frac{1}{2}\left[\frac{d_i L_0}{D}-\delta_{j k} \frac{d_i L_j}{D}-\delta_{i j} \delta_{i k} L_i\right]

        where

        .. math::

            d_i=\frac{1}{\psi_1\left(\alpha_i\right)}, \quad L_i=\partial_i \log d_i, \quad L_0=\partial_{\alpha_0} \log d_0, \quad D=d_0-\sum_i d_i


        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        christoffels : array-like, shape=[..., dim, dim, dim]
            Christoffel symbols, with the contravariant index on
            the first dimension.
            :math:`christoffels[..., i, j, k] = Gamma^i_{jk}`

        References
        ----------
        .. [LPP2021] A. Le Brigant, S. C. Preston, S. Puechmorel. Fisher-Rao
            geometry of Dirichlet Distributions. Differential Geometry
            and its Applications, 74, 101702, 2021.
        """
        dim = self._space.dim
        param = base_point

        param_sum = gs.sum(param, axis=-1, keepdims=True)

        trigamma_param = gs.polygamma(1, param)
        trigamma_sum = gs.polygamma(1, param_sum)

        tetragamma_param = gs.polygamma(2, param)
        tetragamma_sum = gs.polygamma(2, param_sum)

        inv_trigamma_param = 1 / trigamma_param
        inv_trigamma_sum = 1 / trigamma_sum

        d_inv_trigamma_param = -tetragamma_param / trigamma_param**2
        d_inv_trigamma_sum = -tetragamma_sum / trigamma_sum**2

        d_log_inv_trigamma_param = d_inv_trigamma_param / inv_trigamma_param
        d_log_inv_trigamma_sum = d_inv_trigamma_sum / inv_trigamma_sum

        inv_metric_denominator = inv_trigamma_sum - gs.sum(
            inv_trigamma_param, axis=-1, keepdims=True
        )

        eye = gs.eye(dim)

        christoffels_1 = (
            inv_trigamma_param * d_log_inv_trigamma_sum / inv_metric_denominator
        )[..., :, None, None]

        christoffels_2 = (
            -gs.einsum(
                "...i,...j,jk->...ijk",
                inv_trigamma_param,
                d_log_inv_trigamma_param,
                eye,
            )
            / inv_metric_denominator[..., None, None]
        )

        christoffels_3 = -gs.einsum(
            "...i,ij,ik->...ijk",
            d_log_inv_trigamma_param,
            eye,
            eye,
        )

        return 1 / 2 * (christoffels_1 + christoffels_2 + christoffels_3)

    def jacobian_christoffels(self, base_point):
        r"""Compute the Jacobian of the Christoffel symbols.

        Compute the Jacobian of the Christoffel symbols of the
        Fisher information metric.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        jac : array-like, shape=[..., dim, dim, dim, dim]
            Jacobian of the Christoffel symbols.
            :math:`jac[..., i, j, k, l] = dGamma^i_{jk} / dx_l`
        """
        dim = self._space.dim
        param = base_point

        param_sum = gs.sum(param, axis=-1, keepdims=True)

        trigamma_param = gs.polygamma(1, param)
        trigamma_sum = gs.polygamma(1, param_sum)

        tetragamma_param = gs.polygamma(2, param)
        tetragamma_sum = gs.polygamma(2, param_sum)

        pentagamma_param = gs.polygamma(3, param)
        pentagamma_sum = gs.polygamma(3, param_sum)

        inv_trigamma_param = 1 / trigamma_param
        inv_trigamma_sum = 1 / trigamma_sum

        d_inv_trigamma_param = -tetragamma_param / trigamma_param**2
        d_inv_trigamma_sum = -tetragamma_sum / trigamma_sum**2

        d_log_inv_trigamma_param = d_inv_trigamma_param / inv_trigamma_param
        d_log_inv_trigamma_sum = d_inv_trigamma_sum / inv_trigamma_sum

        dd_log_inv_trigamma_param = (
            tetragamma_param**2 - trigamma_param * pentagamma_param
        ) / trigamma_param**2
        dd_log_inv_trigamma_sum = (
            tetragamma_sum**2 - trigamma_sum * pentagamma_sum
        ) / trigamma_sum**2

        inv_metric_denominator = inv_trigamma_sum - gs.sum(
            inv_trigamma_param, axis=-1, keepdims=True
        )

        grad_inv_metric_denominator = d_inv_trigamma_sum - d_inv_trigamma_param

        eye = gs.eye(dim)

        jac_1 = inv_trigamma_param * dd_log_inv_trigamma_sum / inv_metric_denominator
        jac_1_mat = jac_1[..., :, None, None, None]

        jac_2 = (-d_log_inv_trigamma_sum / inv_metric_denominator**2)[
            ..., None
        ] * gs.einsum(
            "...i,...l->...il",
            inv_trigamma_param,
            grad_inv_metric_denominator,
        )
        jac_2_mat = jac_2[..., :, None, None, :]

        jac_3 = d_inv_trigamma_param * d_log_inv_trigamma_sum / inv_metric_denominator
        jac_3_mat = from_vector_to_diagonal_matrix(jac_3)[..., :, None, None, :]

        jac_4_mat = (1 / inv_metric_denominator**2)[..., None, None, None] * gs.einsum(
            "...i,...j,...l,jk->...ijkl",
            inv_trigamma_param,
            d_log_inv_trigamma_param,
            grad_inv_metric_denominator,
            eye,
        )

        jac_5_mat = (
            -gs.einsum(
                "...i,...j,jk,jl->...ijkl",
                inv_trigamma_param,
                dd_log_inv_trigamma_param,
                eye,
                eye,
            )
            / inv_metric_denominator[..., None, None, None]
        )

        jac_6_mat = (
            -gs.einsum(
                "...i,...j,jk,il->...ijkl",
                d_inv_trigamma_param,
                d_log_inv_trigamma_param,
                eye,
                eye,
            )
            / inv_metric_denominator[..., None, None, None]
        )

        jac_7_mat = -gs.einsum(
            "...i,ij,ik,il->...ijkl",
            dd_log_inv_trigamma_param,
            eye,
            eye,
            eye,
        )

        jac = (
            1
            / 2
            * (
                jac_1_mat
                + jac_2_mat
                + jac_3_mat
                + jac_4_mat
                + jac_5_mat
                + jac_6_mat
                + jac_7_mat
            )
        )
        return jac

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        This is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base point onto
        its image.
        In the case of the hyperbolic space, it does not depend on the base point and
        is infinite everywhere, because of the negative curvature.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)


class DirichletRandomVariable(ScipyMultivariateRandomVariable):
    """A Dirichlet random variable."""

    def __init__(self, space):
        pdf = lambda x, point: dirichlet.pdf(gs.transpose(x), point)
        super().__init__(space, dirichlet.rvs, pdf)
