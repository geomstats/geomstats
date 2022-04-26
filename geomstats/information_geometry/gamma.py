"""Statistical Manifold of Gamma distributions with the Fisher metric.

Lead author: Jules Deschamps.
"""
from scipy.stats import gamma

import geomstats.backend as gs
import geomstats.errors
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric

N_STEPS = 100


class GammaDistributions(OpenSet):
    """Class for the manifold of Dirichlet distributions.

    This is :math: Gamma = `(R_+^*)^2`, the positive quadrant of the
    2-dimensional Euclidean space.
    """

    def __init__(self):
        super(GammaDistributions, self).__init__(
            dim=2, ambient_space=Euclidean(2), metric=GammaMetric()
        )

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
            Boolean indicating whether point represents a Dirichlet
            distribution.
        """
        point = gs.array(point, gs.float32)
        point_dim = point.shape[-1]
        belongs = point_dim == 2
        belongs = gs.logical_and(belongs, gs.all(point >= atol, axis=-1))
        return belongs

    def random_point(self, n_samples=1, bound=5.0):
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
        size = (2,) if n_samples == 1 else (n_samples, 2)
        return bound * gs.random.rand(*size)

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
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=2)
        samples = []
        for param in point:
            sample = gs.array(
                gamma.rvs(param[0], loc=0, scale=1 / param[1], size=n_samples)
            )
            samples.append(sample)
        return gs.squeeze(samples[0]) if len(point) == 1 else gs.stack(samples)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the Gamma
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a gamma distribution.

        Returns
        -------
        pdf : function
            Probability density function of the Gamma distribution with
            parameters provided by point.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=2)

        def pdf(x):
            """Generate parameterized function for normal pdf.

            Parameters
            ----------
            x : array-like, shape=[n_points, dim]
                Points of the simplex at which to compute the probability
                density function.

            Returns
            -------
            pdf_at_x : array-like, shape=[..., n_points]
                Values of pdf at x for each value of the parameters provided
                by point.
            """
            pdf_at_x = gs.array(
                [
                    gamma.pdf(t, a=point[:, 0], scale=1 / point[:, 1])
                    for t in gs.array(x)
                ]
            )
            pdf_at_x = gs.squeeze(gs.stack(pdf_at_x, axis=0)).transpose()

            return pdf_at_x

        return pdf


class GammaMetric(RiemannianMetric):
    """Class for the Fisher information metric on Gamma distributions."""

    def __init__(self):
        super(GammaMetric, self).__init__(dim=2)

    def var_change_point(self, point):
        """Compute change of variable of a point.

        Change of variable of a point given in natural coordinates (i.e., (kappa, nu))
        into (kappa, gamma) = (kappa, kappa/nu). Or the opposite change of variable
        depending on which system the point is input, as the change of variable is
        symmetric.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point of the Gamma manifold.

        Returns
        -------
        point : array-like, shape=[..., 2]
            Point of the Gamma manifold.
        """
        point = gs.to_ndarray(point, to_ndim=2)

        point[:, 1] = point[:, 0] / point[:, 1]

        return gs.squeeze(point)

    def var_change_vec(self, vec, base_point):
        """Compute change of variable of a tangent vector.

        Change of variable of a tangent vector at base point, both given in the same
        system (i.e., (kappa, scale)) into (kappa, scale') = (kappa, kappa/scale).
        Again, the change of variable is symmetric.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point of the Gamma manifold.
        vec : array-like, shape=[..., 2]
            Tangent vector at base_point

        Returns
        -------
        vec : array-like, shape=[..., 2]
            Tangent vector at base_point
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        n_points = base_point.shape[0]

        kappa, scale = (
            base_point[:, 0],
            base_point[:, 1],
        )  # scale is either nu or gamma, works in both ways!

        jac = gs.array(
            [[gs.ones(n_points), gs.zeros(n_points)], [1 / scale, -kappa / scale**2]]
        )
        jac = gs.transpose(jac, [2, 0, 1])

        vec = vec.reshape((n_points, 2, 1))
        vec = gs.matmul(jac, vec)

        return gs.squeeze(vec)

    def metric_matrix(self, base_point=None):
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
        if base_point is None:
            raise ValueError(
                "A base point must be given to compute the " "metric matrix"
            )
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        kappa, gamma = base_point[:, 0], base_point[:, 1]

        mat_diag = gs.transpose(
            gs.array([kappa / gamma**2, gs.polygamma(1, kappa) - 1 / kappa])
        )
        mat = from_vector_to_diagonal_matrix(mat_diag)
        return gs.squeeze(mat)

    def christoffels(self, base_point):
        """Compute the Christoffel symbols, when given in simple coordinates.

        Compute the Christoffel symbols of the Fisher information metric.

        References
        ----------
        Arwini, K. A., & Dodson, C. T. (2008). Information geometry (pp. 31-54).
        Springer Berlin Heidelberg.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        christoffels : array-like, shape=[..., 2, 2, 2]
            Christoffel symbols, with the contravariant index on
            the first dimension.
            :math: 'christoffels[..., i, j, k] = Gamma^i_{jk}'
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        kappa, gamma = base_point[:, 0], base_point[:, 1]

        shape = kappa.shape

        c1 = gs.squeeze(
            from_vector_to_diagonal_matrix(
                gs.transpose(
                    gs.array(
                        [
                            (kappa**2 * gs.polygamma(2, kappa) + 1)
                            / (2 * (kappa**2 * gs.polygamma(1, kappa) - kappa)),
                            -kappa
                            / (2 * gamma**2 * (kappa * gs.polygamma(1, kappa) - 1)),
                        ]
                    )
                )
            )
        )

        c2 = gs.squeeze(
            gs.transpose(
                gs.array(
                    [[gs.zeros(shape), 1 / (2 * kappa)], [1 / (2 * kappa), -1 / gamma]]
                )
            )
        )

        christoffels = gs.array([c1, c2])

        if len(christoffels.shape) > 3:
            christoffels = gs.stack(christoffels, 1)

        return gs.squeeze(christoffels)

    def jacobian_christoffels(self, base_point):
        """Compute the Jacobian of the Christoffel symbols, when given in simple coordinates.

        Compute the Jacobian of the Christoffel symbols of the
        Fisher information metric.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        jac : array-like, shape=[..., 2, 2, 2, 2]
            Jacobian of the Christoffel symbols.
            :math: 'jac[..., i, j, k, l] = dGamma^i_{jk} / dx_l'
        """
        base_point = gs.to_ndarray(base_point, 2)

        n_points = base_point.shape[0]

        kappa, gamma = base_point[:, 0], base_point[:, 1]

        term_0 = gs.zeros((n_points))
        term_1 = 1 / gamma**2
        term_2 = kappa / (gamma**3 * (kappa * gs.polygamma(1, kappa) - 1))
        term_3 = -1 / (2 * kappa**2)
        term_4 = (kappa**2 * gs.polygamma(2, kappa) + 1) / (
            2 * gamma**2 * (kappa * gs.polygamma(1, kappa) - 1) ** 2
        )
        term_5 = (
            kappa**4
            * (
                gs.polygamma(1, kappa) * gs.polygamma(3, kappa)
                - gs.polygamma(2, kappa) ** 2
            )
            - kappa**3 * gs.polygamma(3, kappa)
            - 2 * kappa**2 * gs.polygamma(2, kappa)
            - 2 * kappa * gs.polygamma(1, kappa)
            + 1
        ) / (2 * (kappa**2 * gs.polygamma(1, kappa) - kappa) ** 2)

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
