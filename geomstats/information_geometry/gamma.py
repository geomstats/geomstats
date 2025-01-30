"""Statistical Manifold of Gamma distributions with the Fisher metric.

The natural coordinate system for a Gamma Distribution is:
point = [kappa, nu], where kappa is the shape parameter, and nu the rate, or 1/scale.

However, information geometry most often works with standard coordinates, given by:
point = [kappa, gamma] = [kappa, kappa/nu].

The standard coordinate system is the convention we use in this script.
All points and all vectors input are assumed to be given in the standard coordinate
system unless stated otherwise.

Some of the methods in GammaDistributions allow to easily make the associated
change of variable, either for a point or a vector.

Lead author: Jules Deschamps.
"""

from scipy.stats import gamma

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.base import VectorSpaceOpenSet
from geomstats.geometry.diffeo import InvolutionDiffeomorphism
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyUnivariateRandomVariable,
)
from geomstats.numerics.bvp import ScipySolveBVP
from geomstats.numerics.geodesic import ExpODESolver, LogODESolver
from geomstats.numerics.ivp import ScipySolveIVP
from geomstats.vectorization import get_batch_shape


class NaturalToStandardDiffeo(InvolutionDiffeomorphism):
    """Diffeomorphism between natural and standard coordinates."""

    def __call__(self, point):
        """Convert point from natural coordinates to standard coordinates.

        The change of variable is symmetric.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point of the Gamma manifold, given in natural coordinates.

        Returns
        -------
        point : array-like, shape=[..., 2]
            Point of the Gamma manifold, given in standard coordinates.
        """
        return gs.stack([point[..., 0], point[..., 0] / point[..., 1]], axis=-1)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Convert tangent vector from natural coordinates to standard coordinates.

        The change of variable is symmetric.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 2]
            Tangent vector at base_point, given in natural coordinates.
        base_point : array-like, shape=[..., 2]
            Point of the Gamma manifold, given in natural coordinates.
        image_point : array-like, shape=[..., 2]
            Point of the Gamma manifold, given in standard coordinates.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 2]
            Tangent vector at base_point, given in standard coordinates.
        """
        if base_point is None:
            base_point = self(image_point)

        kappa, scale = base_point[..., 0], base_point[..., 1]

        jac_row_1 = gs.array([1.0, 0.0])
        jac_row_2 = gs.stack([1 / scale, -kappa / scale**2], axis=-1)

        point_batch_shape = get_batch_shape(1, base_point)
        if point_batch_shape:
            jac_row_1 = gs.broadcast_to(jac_row_1, point_batch_shape + (2,))

        jac = gs.stack([jac_row_1, jac_row_2], axis=-2)

        return gs.einsum("...jk,...k->...j", jac, tangent_vec)


class GammaDistributions(InformationManifoldMixin, VectorSpaceOpenSet):
    """Class for the manifold of Gamma distributions.

    This is :math:`Gamma = (R_+^*)^2`, the positive quadrant of the
    2-dimensional Euclidean space.
    """

    def __init__(self, equip=True):
        super().__init__(
            dim=2,
            embedding_space=Euclidean(2, equip=False),
            support_shape=(),
            equip=equip,
        )

        self._scp_rv = GammaDistributionsRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return GammaMetric

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold of Gamma distributions.

        Check that point defines parameters for a Gamma distribution,
        i.e. belongs to the positive quadrant of the Euclidean space.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point to be checked.
        atol : float
            Tolerance to evaluate positivity.
            Optional, default: gs.atol

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean indicating whether point represents a Gamma
            distribution.
        """
        point_dim = point.shape[-1]
        belongs = point_dim == 2
        return gs.logical_and(belongs, gs.all(point >= -atol, axis=-1))

    def random_point(self, n_samples=1, upper_bound=5.0, lower_bound=0.0):
        """Sample parameters of Gamma distributions.

        The uniform distribution on [0, bound]^2 is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of the square where the Gamma parameters are sampled.
            Optional, default: 5.

        Returns
        -------
        samples : array-like, shape=[..., 2]
            Sample of points representing Gamma distributions.
        """
        upper_bound = upper_bound * gs.ones(2)
        lower_bound = lower_bound * gs.ones(2)

        if gs.any((upper_bound - lower_bound) < 0):
            raise ValueError("upper_bound cannot be greater than lower_bound.")

        size = (2,) if n_samples == 1 else (n_samples, 2)
        return lower_bound + (upper_bound - lower_bound) * gs.random.rand(*size)

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the open set.

        The last coordinate is floored to `gs.atol` if it is negative.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point in ambient space.
        atol : float
            Tolerance to evaluate positivity.

        Returns
        -------
        projected : array-like, shape=[..., 2]
            Projected point.
        """
        return gs.where(point < atol, atol, point)

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
            Optional, default: 1.

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


class GammaMetric(RiemannianMetric):
    """Class for the Fisher information metric on Gamma distributions.

    References
    ----------
    .. [AD2008] Arwini, K. A., & Dodson, C. T. (2008).
        Information geometry (pp. 31-54). Springer Berlin Heidelberg.
    """

    def __init__(self, space):
        super().__init__(space=space)

        self.log_solver = LogODESolver(
            space, n_nodes=1000, integrator=ScipySolveBVP(max_nodes=1000)
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

    def christoffels(self, base_point):
        """Compute the Christoffel symbols.

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
        """Compute the Jacobian of the Christoffel symbols.

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
        base_point = gs.to_ndarray(base_point, 2)

        n_points = base_point.shape[0]

        kappa, gamma = base_point[:, 0], base_point[:, 1]

        term_0 = gs.zeros((n_points))
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

        if n_points > 1:
            jac = gs.transpose(jac, [4, 0, 1, 2, 3])

        return gs.squeeze(jac)


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
