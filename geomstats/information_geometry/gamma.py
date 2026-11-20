"""Statistical Manifold of Gamma distributions with the Fisher metric.

A Gamma distribution is commonly parametrized by its shape and rate:
point = [kappa, nu], where kappa is the shape parameter, and nu the rate, or 1/scale.

However, information geometry most often works with its shape and mean:
point = [kappa, gamma] = [kappa, kappa/nu].

The shape-mean parametrization is the convention we use in this script.
All points and all vectors input are assumed to be given in the shape-mean
parametrization unless stated otherwise.

``ScaledReciprocalDiffeo`` makes the associated change of variable, either
for a point or a vector.

Lead author: Jules Deschamps.
"""

import math

from scipy.stats import gamma

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.positive_orthant import PositiveOrthant, ScaledReciprocalDiffeo
from geomstats.geometry.pullback_metric import CoordinatePullbackDiffeoMetric
from geomstats.geometry.riemannian_metric import CoordinateRiemannianMetric
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyUnivariateRandomVariable,
)
from geomstats.numerics.bvp import ScipySolveBVP
from geomstats.numerics.geodesic import ExpODESolver, LogODESolver
from geomstats.numerics.ivp import ScipySolveIVP
from geomstats.vectorization import repeat_out


class GammaDistributions(InformationManifoldMixin, PositiveOrthant):
    r"""Class for the manifold of Gamma distributions.

    This is :math:`Gamma = (R_+^*)^2`, the positive quadrant of the
    2-dimensional Euclidean space.
    """

    def __init__(self, equip=True):
        super().__init__(dim=2, support_shape=(), equip=equip)
        self._scp_rv = GammaDistributionsRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return GammaMetric

    def sample(self, point, n_samples=1):
        """Sample from the Gamma distribution.

        Sample from the Gamma distribution with parameters provided
        by point. This gives n_samples points.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a Gamma distribution.
        n_samples : int
            Number of points to sample for each set of parameters in point.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from the Gamma distributions.
        """
        return self._scp_rv.rvs(point, n_samples)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the Gamma
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a Gamma distribution.

        Returns
        -------
        pdf : function
            Probability density function of the Gamma distribution with
            parameters provided by point.
        """
        kappa = gs.expand_dims(point[..., 0], axis=-1)
        gamma = gs.expand_dims(point[..., 1], axis=-1)

        def pdf(x):
            """Generate parameterized function for Gamma pdf.

            Parameters
            ----------
            x : array-like, shape=[n_samples,]
                Points at which to compute the probability
                density function.

            Returns
            -------
            pdf_at_x : array-like, shape=[..., n_samples]
                Values of pdf at x for each value of the parameters provided
                by point.
            """
            x = gs.reshape(gs.array(x), (-1,))
            return (
                kappa**kappa
                * x ** (kappa - 1)
                * gs.exp(-kappa * x / gamma)
                / (gamma**kappa * gs.gamma(kappa))
            )

        return pdf

    @staticmethod
    def maximum_likelihood_fit(data):
        """Estimate parameters from samples.

        This is a wrapper around scipy's maximum likelihood estimator to
        estimate the parameters of a gamma distribution from samples.

        Parameters
        ----------
        data : list or list of lists/arrays
            Data to estimate parameters from. Lists of
            different length may be passed.

        Returns
        -------
        parameter : array-like, shape=[..., 2]
            Estimate of parameter obtained by maximum likelihood.
        """

        def is_nested(sample):
            """Check if sample contains an iterable."""
            for el in sample:
                try:
                    return iter(el)
                except TypeError:
                    return False

        if not is_nested(data):
            data = [data]
        parameters = []
        for sample in data:
            sample = gs.array(sample)
            kappa, _, scale = gamma.fit(sample, floc=0)
            nu = 1 / scale
            parameters.append(gs.array([kappa, kappa / nu]))
        return parameters[0] if len(data) == 1 else gs.stack(parameters)


class ShapeRateGammaDistributions(PositiveOrthant):
    r"""Manifold of Gamma distributions in the shape-rate parametrization.

    Points are given by :math:`[\kappa, \nu]`, where :math:`\kappa` is the
    shape parameter and :math:`\nu` the rate, or 1/scale. They are mapped to
    the shape-mean parametrization of :class:`GammaDistributions` by
    :class:`ScaledReciprocalDiffeo`.

    Parameters
    ----------
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, equip=True):
        super().__init__(dim=2, equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return ShapeRateGammaMetric


class GammaMetric(CoordinateRiemannianMetric):
    """Class for the Fisher information metric on Gamma distributions.

    References
    ----------
    .. [AD2008] Arwini, K. A., & Dodson, C. T. (2008).
        Information geometry (pp. 31-54). Springer Berlin Heidelberg.
    """

    def __init__(self, space):
        super().__init__(space=space)

        self.log_solver = LogODESolver(
            space,
            n_nodes=10,
            integrator=ScipySolveBVP(max_nodes=1000),
        )
        self.exp_solver = ExpODESolver(space, integrator=ScipySolveIVP(method="LSODA"))

    def metric_matrix(self, base_point):
        """Compute the inner-product matrix.

        Compute the inner-product matrix of the Fisher information metric
        at the tangent space at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., 2, 2]
            Inner-product matrix.
        """
        kappa, gamma = base_point[..., 0], base_point[..., 1]
        mat_diag = gs.stack(
            [gs.polygamma(1, kappa) - 1 / kappa, kappa / gamma**2], axis=-1
        )
        return from_vector_to_diagonal_matrix(mat_diag)

    def inner_product_derivative_matrix(self, base_point):
        r"""Compute derivative of the inner prod matrix at base point.

        The non-zero entries are
        :math:`\partial_\kappa g_{\kappa\kappa} = \psi''(\kappa) + 1 / \kappa^2`,
        :math:`\partial_\kappa g_{\gamma\gamma} = 1 / \gamma^2` and
        :math:`\partial_\gamma g_{\gamma\gamma} = -2 \kappa / \gamma^3`,
        where :math:`\psi` is the digamma function.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        metric_derivative : array-like, shape=[..., 2, 2, 2]
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: :math:`mat_{ijk} = \partial_k g_{ij}`.
        """
        kappa, gamma = base_point[..., 0], base_point[..., 1]
        derivative_kappa = from_vector_to_diagonal_matrix(
            gs.stack([gs.polygamma(2, kappa) + 1 / kappa**2, 1 / gamma**2], axis=-1)
        )
        derivative_gamma = from_vector_to_diagonal_matrix(
            gs.stack([gs.zeros_like(kappa), -2 * kappa / gamma**3], axis=-1)
        )
        return gs.stack([derivative_kappa, derivative_gamma], axis=-1)

    def christoffels(self, base_point):
        r"""Compute the Christoffel symbols.

        Compute the Christoffel symbols of the Fisher information metric.
        For computation purposes, we replace the value of
        (gs.polygamma(1, x) - 1/x) by an equivalent (close lower-bound) when it becomes
        too difficult to compute, as per in [GQ2015]_.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        christoffels : array-like, shape=[..., 2, 2, 2]
            Christoffel symbols, with the contravariant index on
            the first dimension.
            :math:`christoffels[..., i, j, k] = Gamma^i_{jk}`

        References
        ----------
        .. [AD2008] Arwini, K. A., & Dodson, C. T. (2008).
            Information geometry (pp. 31-54). Springer Berlin Heidelberg.
        .. [GQ2015] Guo, B. N., Qi, F., Zhao, J. L., & Luo, Q. M. (2015).
            Sharp inequalities for polygamma functions.
            Mathematica Slovaca, 65(1), 103-120.
        """
        kappa, gamma = base_point[..., 0], base_point[..., 1]

        if gs.any(kappa > 4e15):
            raise ValueError(
                "Christoffels computation overflows with values of kappa. "
                "All values of kappa < 4e15 work."
            )

        c111 = gs.where(
            gs.polygamma(1, kappa) - 1 / kappa > gs.atol,
            (gs.polygamma(2, kappa) + gs.array(kappa) ** -2)
            / (2 * (gs.polygamma(1, kappa) - 1 / kappa)),
            0.25 * (kappa**2 * gs.polygamma(2, kappa) + 1),
        )

        c122 = gs.where(
            gs.polygamma(1, kappa) - 1 / kappa > gs.atol,
            -1 / (2 * gamma**2 * (gs.polygamma(1, kappa) - 1 / kappa)),
            -(kappa**2) / (4 * gamma**2),
        )

        c1 = from_vector_to_diagonal_matrix(gs.stack([c111, c122], axis=-1))
        c2 = gs.stack(
            [
                gs.stack([gs.zeros_like(kappa), 1 / (2 * kappa)], axis=-1),
                gs.stack([1 / (2 * kappa), -1 / gamma], axis=-1),
            ],
            axis=-1,
        )

        return gs.stack([c1, c2], axis=-3)

    def jacobian_christoffels(self, base_point):
        r"""Compute the Jacobian of the Christoffel symbols.

        Compute the Jacobian of the Christoffel symbols of the
        Fisher information metric.

        For computation purposes, we replace the value of
        (gs.polygamma(1, x) - 1/x) and (gs.polygamma(2,x) + 1/x**2) by an equivalent
        (close bounds) when they become too difficult to compute.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        jac : array-like, shape=[..., 2, 2, 2, 2]
            Jacobian of the Christoffel symbols.
            :math:`jac[..., i, j, k, l] = dGamma^i_{jk} / dx_l`

        References
        ----------
        .. [GQ2015] Guo, B. N., Qi, F., Zhao, J. L., & Luo, Q. M. (2015).
            Sharp inequalities for polygamma functions.
            Mathematica Slovaca, 65(1), 103-120.
        """
        batch_shape = base_point.shape[:-1]

        kappa, gamma = base_point[..., 0], base_point[..., 1]

        term_0 = gs.zeros(batch_shape)
        term_1 = 1 / gamma**2
        term_2 = gs.where(
            gs.polygamma(1, kappa) - 1 / kappa > gs.atol,
            kappa / (gamma**3 * (kappa * gs.polygamma(1, kappa) - 1)),
            kappa**2 / gamma**3,
        )
        term_3 = -1 / (2 * kappa**2)
        term_4 = gs.where(
            gs.polygamma(1, kappa) - 1 / kappa > gs.atol,
            (kappa**2 * gs.polygamma(2, kappa) + 1)
            / (2 * gamma**2 * (kappa * gs.polygamma(1, kappa) - 1) ** 2),
            (kappa**4 * gs.polygamma(2, kappa) + kappa**2) / (2 * gamma**2),
        )
        term_5 = gs.where(
            gs.polygamma(1, kappa) - 1 / kappa > gs.atol,
            (
                kappa**4
                * (
                    gs.polygamma(1, kappa) * gs.polygamma(3, kappa)
                    - gs.polygamma(2, kappa) ** 2
                )
                - kappa**3 * gs.polygamma(3, kappa)
                - 2 * kappa**2 * gs.polygamma(2, kappa)
                - 2 * kappa * gs.polygamma(1, kappa)
                + 1
            )
            / (2 * (kappa**2 * gs.polygamma(1, kappa) - kappa) ** 2),
            0.5
            * (
                kappa**4
                * (
                    gs.polygamma(1, kappa) * gs.polygamma(3, kappa)
                    - gs.polygamma(2, kappa) ** 2
                )
                - kappa**3 * gs.polygamma(3, kappa)
                - 2 * kappa**2 * gs.polygamma(2, kappa)
                - 2 * kappa * gs.polygamma(1, kappa)
                + 1
            ),
        )

        jac = gs.array(
            [
                [
                    [[term_5, term_0], [term_0, term_0]],
                    [[term_0, term_0], [term_4, term_2]],
                ],
                [
                    [[term_0, term_0], [term_3, term_0]],
                    [[term_3, term_0], [term_0, term_1]],
                ],
            ]
        )

        n_batch_axes = len(batch_shape)
        if n_batch_axes:
            jac = gs.moveaxis(
                jac, tuple(range(-n_batch_axes, 0)), tuple(range(n_batch_axes))
            )

        return jac

    def riemann_tensor(self, base_point):
        r"""Compute Riemannian tensor at base_point.

        In dimension two, the curvature is determined by the Gaussian
        curvature :math:`K`:
        :math:`R(X, Y)Z = K (\langle Y, Z \rangle X - \langle X, Z \rangle Y)`,
        i.e. :math:`R_{ijk}^l = K (g_{ki} \delta_j^l - g_{ji} \delta_k^l)`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        riemann_curvature : array-like, shape=[..., 2, 2, 2, 2]
            riemann_tensor[...,i,j,k,l] = R_{ijk}^l
            Riemannian tensor curvature,
            with the contravariant index on the last dimension.
        """
        metric_mat = self.metric_matrix(base_point)
        eye = gs.eye(2, dtype=metric_mat.dtype)
        constant_curvature_tensor = gs.einsum(
            "...ki,jl->...ijkl", metric_mat, eye
        ) - gs.einsum("...ji,kl->...ijkl", metric_mat, eye)
        return gs.einsum(
            "...,...ijkl->...ijkl",
            self.gaussian_curvature(base_point),
            constant_curvature_tensor,
        )

    def sectional_curvature(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the sectional curvature.

        In dimension two, the sectional curvature is the Gaussian curvature,
        whatever the tangent vectors (including collinear ones, for which
        the general definition is degenerate).

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., 2]
            Tangent vector at ``base_point``.
        tangent_vec_b : array-like, shape=[..., 2]
            Tangent vector at ``base_point``.
        base_point : array-like, shape=[..., 2]
            Point in the manifold.

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at ``base_point``.
        """
        return repeat_out(
            self._space.point_ndim,
            self.gaussian_curvature(base_point),
            tangent_vec_a,
            tangent_vec_b,
            base_point,
        )

    def scalar_curvature(self, base_point):
        """Compute scalar curvature at base_point.

        The space is two-dimensional, so the scalar curvature is twice the
        Gaussian curvature.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        curvature : array-like, shape=[...,]
            Scalar curvature.
        """
        return 2 * self.gaussian_curvature(base_point)

    def gaussian_curvature(self, base_point):
        r"""Compute the Gaussian curvature.

        The Gaussian curvature of the Fisher information metric only depends
        on :math:`\kappa`:

        .. math::

            K = \frac{\psi'(\kappa) + \kappa \psi''(\kappa)}
            {4 (\kappa \psi'(\kappa) - 1)^2}

        where :math:`\psi` is the digamma function.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        curvature : array-like, shape=[...,]
            Gaussian curvature at `base_point`.
        """
        kappa = base_point[..., 0]
        return (gs.polygamma(1, kappa) + kappa * gs.polygamma(2, kappa)) / (
            4 * (-1 + kappa * gs.polygamma(1, kappa)) ** 2
        )

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        The space is simply connected, complete and has negative curvature,
        so it is infinite.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)


class ShapeRateGammaMetric(CoordinatePullbackDiffeoMetric):
    """Fisher information metric on Gamma distributions in shape-rate form.

    It is the pullback of :class:`GammaMetric` through
    :class:`ScaledReciprocalDiffeo`.
    """

    def __init__(self, space):
        super().__init__(
            space=space,
            diffeo=ScaledReciprocalDiffeo(),
            image_space=GammaDistributions(),
        )


class GammaDistributionsRandomVariable(ScipyUnivariateRandomVariable):
    """A gamma random variable."""

    def __init__(self, space):
        super().__init__(space, gamma.rvs, gamma.pdf)

    @staticmethod
    def _flatten_params(point, pre_flat_shape):
        param_a = gs.expand_dims(point[..., 0], axis=-1)
        scale = gs.expand_dims(point[..., 1] / point[..., 0], axis=-1)

        flat_param_a = gs.reshape(gs.broadcast_to(param_a, pre_flat_shape), (-1,))
        flat_scale = gs.reshape(gs.broadcast_to(scale, pre_flat_shape), (-1,))
        return {"a": flat_param_a, "scale": flat_scale}
