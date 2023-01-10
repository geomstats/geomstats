"""Statistical Manifold of geometric distributions with the Fisher metric.

Lead author: Tra My Nguyen.
"""

from scipy.stats import geom

import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.information_geometry.base import InformationManifoldMixin


class GeometricDistributions(InformationManifoldMixin, OpenSet):
    """Class for the manifold of geometric distributions.

    This is the parameter space of geometric distributions
    i.e. the half-line of positive reals.
    """

    def __init__(self):
        super().__init__(
            dim=1,
            embedding_space=Euclidean(dim=1),
            metric=GeometricMetric(),
        )

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold of geometric distributions.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point to be checked.
        atol : float
            Tolerance to evaluate positivity.
            Optional, default: gs.atol

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean indicating whether point represents an geometric
            distribution.
        """
        return gs.squeeze(gs.logical_and(atol <= point, point <= 1 - atol))

    def random_point(self, n_samples=1, lower_bound=0.1):
        """Sample parameters of Possion distributions.

        The uniform distribution on (lower_bound, 1) is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Right-end ot the segment where geometric parameters are sampled.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[n_samples,]
            Sample of points representing geometric distributions.
        """
        size = (n_samples, self.dim) if n_samples != 1 else (self.dim,)
        return (1-lower_bound) * gs.random.rand(*size) + lower_bound

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the open set.

        The last coordinate is floored to `gs.atol` if it is non-positive.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point in ambient space.
        atol : float
            Tolerance to evaluate positivity.

        Returns
        -------
        projected : array-like, shape=[..., 1]
            Projected point.
        """
        return gs.where(
            gs.logical_or(point < atol, point > 1 - atol),
            (1 - atol) * gs.cast(point > 1 - atol, point.dtype)
            + atol * gs.cast(point < atol, point.dtype),
            point,
        )

    def sample(self, point, n_samples=1):
        """Sample from the geometric distribution.

        Sample from the geometric distribution with parameter provided
        by point.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point representing an geometric distribution.
        n_samples : int
            Number of points to sample with each parameter in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from geometric distributions.
        """

        def _sample(param):
            return geom.rvs(param, size=n_samples)

        if point.ndim > 1:
            return gs.array([_sample(point_) for point_ in point])

        return gs.array(_sample(point))

    def point_to_pdf(self, point):
        """Compute pmf associated to point.

        Compute the probability mass function of the geometric
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point representing an geometric distribution (scale).

        Returns
        -------
        pdf : function
            Probability mass function of the geometric distribution with
            scale parameter provided by point.
        """

        def pmf(k):
            """Generate parameterized function for geometric pmf.

            Compute the probability mass function of the geometric
            distribution with parameters provided by point.

            Parameters
            ----------
            k : array-like, shape=[n_points,]
                Point representing an geometric distribution (lambda).

            Returns
            -------
            pmf_at_k : array-like, shape=[..., n_points]
                Probability mass function of the geometric distribution with
                parameters provided by point.
            """
            k = gs.reshape(gs.array(k), (-1,))
            return (1 - point) ** (k - 1) * point

        return pmf


class GeometricMetric(RiemannianMetric):
    """Class for the Fisher information metric on geometric distributions.

    References
    ----------
    .. [AM1981] Atkinson, C., & Mitchell, A. F. (1981). Rao's distance measure.
        Sankhyā: The Indian Journal of Statistics, Series A, 345-365.
    """

    def __init__(self):
        super().__init__(dim=1)

    def squared_dist(self, point_a, point_b, **kwargs):
        """Compute squared distance associated with the geometric metric.

        Parameters
        ----------
        point_a : array-like, shape=[..., 1]
            Point representing an geometric distribution (lambda parameter).
        point_b : array-like, shape=[..., 1]
            Point representing a geometric distribution (lambda parameter).

        Returns
        -------
        squared_dist : array-like, shape=[...,]
            Squared distance between points point_a and point_b.
        """
        point_a, point_b = gs.broadcast_arrays(point_a, point_b)
        return gs.squeeze(
            4
            * (gs.arctanh(gs.sqrt(1 - point_a)) - gs.arctanh(gs.sqrt(1 - point_b))) ** 2
        )

    def metric_matrix(self, base_point):
        """Compute the metric matrix at the tangent space at base_point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Point representing a geometric distribution.

        Returns
        -------
        mat : array-like, shape=[..., 1, 1]
            Metric matrix.
        """
        return gs.expand_dims(1 / (base_point**2 * (1 - base_point)), axis=-1)

    def _geodesic_ivp(self, initial_point, initial_tangent_vec):
        """Solve geodesic initial value problem.

        Compute the parameterized function for the geodesic starting at
        initial_point with initial velocity given by initial_tangent_vec.

        Parameters
        ----------
        initial_point : array-like, shape=[..., 1]
            Initial point.

        initial_tangent_vec : array-like, shape=[..., 1]
            Tangent vector at initial point.

        Returns
        -------
        path : function
            Parameterized function for the geodesic curve starting at
            initial_point with velocity initial_tangent_vec.
        """
        initial_point = gs.broadcast_to(initial_point, initial_tangent_vec.shape)

        initial_phase = gs.arctanh(gs.sqrt(1 - initial_point))
        frequency = -initial_tangent_vec / (
            2 * initial_point * gs.sqrt(1 - initial_point)
        )

        def path(t):
            """Generate parameterized function for geodesic curve.

            Parameters
            ----------
            t : array-like, shape=[n_times,]
                Times at which to compute points of the geodesics.

            Returns
            -------
            geodesic : array-like, shape=[..., n_times, 1]
                Values of the geodesic at times t.
            """
            return gs.expand_dims(
                1 - gs.tanh(frequency * t + initial_phase) ** 2, axis=-1
            )

        return path

    def _geodesic_bvp(self, initial_point, end_point):
        """Solve geodesic boundary problem.

        Compute the parameterized function for the geodesic starting at
        initial_point and ending at end_point.

        Parameters
        ----------
        initial_point : array-like, shape=[..., 1]
            Initial point.
        end_point : array-like, shape=[..., 1]
            End point.

        Returns
        -------
        path : function
            Parameterized function for the geodesic curve starting at
            initial_point and ending at end_point.
        """
        initial_point, end_point = gs.broadcast_arrays(initial_point, end_point)

        initial_phase = gs.arctanh(gs.sqrt(1 - initial_point))
        frequency = gs.arctanh(gs.sqrt(1 - end_point)) - initial_phase

        def path(t):
            """Generate parameterized function for geodesic curve.

            Parameters
            ----------
            t : array-like, shape=[n_times,]
                Times at which to compute points of the geodesics.

            Returns
            -------
            geodesic : array-like, shape=[..., n_times, 1]
                Values of the geodesic at times t.
            """
            return gs.expand_dims(
                1 - gs.tanh(frequency * t + initial_phase) ** 2, axis=-1
            )

        return path

    def geodesic(
        self, initial_point, end_point=None, initial_tangent_vec=None, **exp_kwargs
    ):
        """Generate parameterized function for the geodesic curve.

        Geodesic curve defined by either:

        - an initial point and an initial tangent vector,
        - an initial point and an end point.

        Parameters
        ----------
        initial_point : array-like, shape=[..., 1]
            Point on the manifold, initial point of the geodesic.
        end_point : array-like, shape=[..., 1], optional
            Point on the manifold, end point of the geodesic. If None,
            an initial tangent vector must be given.
        initial_tangent_vec : array-like, shape=[..., 1],
            Tangent vector at base point, the initial speed of the geodesics.
            Optional, default: None.
            If None, an end point must be given and a logarithm is computed.

        Returns
        -------
        path : callable
            Time parameterized geodesic curve. If a batch of initial
            conditions is passed, the output array's first dimension
            represents time, and the second corresponds to the different
            initial conditions.
        """
        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end point or an initial tangent "
                "vector to define the geodesic."
            )
        if end_point is not None:
            if initial_tangent_vec is not None:
                raise ValueError(
                    "Cannot specify both an end point " "and an initial tangent vector."
                )
            path = self._geodesic_bvp(initial_point, end_point)

        if initial_tangent_vec is not None:
            path = self._geodesic_ivp(initial_point, initial_tangent_vec)

        return path

    def exp(self, tangent_vec, base_point):
        """Compute exp map of a base point in tangent vector direction.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., 1]
            End point of the geodesic starting at base_point with
            initial velocity tangent_vec.
        """
        return (
            1
            - gs.tanh(
                -tangent_vec / (2 * base_point * gs.sqrt(1 - base_point))
                + gs.arctanh(gs.sqrt(1 - base_point))
            )
            ** 2
        )

    def log(self, end_point, base_point):
        """Compute log map using a base point and an end point.

        Parameters
        ----------
        end_point : array-like, shape=[..., 1]
            End point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Initial velocity of the geodesic starting at base_point and
            reaching end_point at time 1.
        """
        return (
            -2
            * base_point
            * gs.sqrt(1 - base_point)
            * (gs.arctanh(gs.sqrt(1 - end_point)) - gs.arctanh(gs.sqrt(1 - base_point)))
        )
