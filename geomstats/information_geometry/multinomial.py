"""Statistical Manifold of multinomial distributions with the Fisher metric.

Lead author: Alice Le Brigant.
"""

from scipy.stats import dirichlet, multinomial

import geomstats.backend as gs
import geomstats.errors
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.base import LevelSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import HypersphereMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.information_geometry.base import InformationManifoldMixin


class MultinomialDistributions(InformationManifoldMixin, LevelSet):
    r"""Class for the manifold of multinomial distributions.

    This is the set of `n+1`-tuples of positive reals that sum up to one,
    i.e. the `n`-simplex. Each point is the parameter of a multinomial
    distribution, i.e. gives the probabilities of $n$ different outcomes
    in a single experiment.

    Attributes
    ----------
    dim : int
        Dimension of the parameter manifold of multinomial distributions.
        The number of outcomes is dim + 1.
    embedding_manifold : Manifold
        Embedding manifold.
    """

    def __init__(self, dim, n_draws, **kwargs):
        super().__init__(
            dim=dim,
            embedding_space=Euclidean(dim + 1),
            submersion=lambda x: gs.sum(x, axis=-1),
            value=1.0,
            tangent_submersion=lambda v, x: gs.sum(v, axis=-1),
        )
        self.n_draws = n_draws
        self.metric = MultinomialMetric(dim=dim, n_draws=n_draws)

    def random_point(self, n_samples=1):
        """Generate parameters of multinomial distributions.

        The Dirichlet distribution on the simplex is used
        to generate parameters.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Sample of points representing multinomial distributions.
        """
        samples = gs.from_numpy(dirichlet.rvs(gs.ones(self.dim + 1), size=n_samples))

        return samples[0] if n_samples == 1 else samples

    def projection(self, point, atol=gs.atol):
        """Project a point on the simplex.

        Negative components are replaced by zero and the point is
        renormalized by its 1-norm.

        Parameters
        ----------
        point: array-like, shape=[..., dim + 1]
            Point in embedding Euclidean space.
         atol : float
            Tolerance to evaluate positivity.

        Returns
        -------
        projected_point : array-like, shape=[..., dim + 1]
            Point projected on the simplex.
        """
        geomstats.errors.check_belongs(point, self.embedding_space)
        point_quadrant = gs.where(point < atol, atol, point)
        norm = gs.sum(point_quadrant, axis=-1)
        projected_point = gs.einsum("...,...i->...i", 1.0 / norm, point_quadrant)
        return projected_point

    def to_tangent(self, vector, base_point=None):
        """Project a vector to the tangent space.

        Project a vector in Euclidean space on the tangent space of
        the simplex at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector in Euclidean space.
        base_point : array-like, shape=[..., dim + 1]
            Point on the simplex defining the tangent space,
            where the vector will be projected.

        Returns
        -------
        vector : array-like, shape=[..., dim + 1]
            Tangent vector in the tangent space of the simplex
            at the base point.
        """
        geomstats.errors.check_belongs(vector, self.embedding_space)
        component_mean = gs.mean(vector, axis=-1)
        return gs.transpose(gs.transpose(vector) - component_mean)

    def sample(self, point, n_samples=1):
        """Sample from the multinomial distribution.

        Sample from the multinomial distribution with parameters provided by
        point. This gives samples in the simplex.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Parameters of a multinomial distribution, i.e. probabilities
            associated to dim + 1 outcomes.
        n_samples : int
            Number of points to sample with each set of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples, dim + 1]
            Samples from multinomial distributions.
            Note that this can be of shape [n_points, n_samples, dim + 1] if
            several points and several samples are provided as inputs.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=2)
        samples = []
        for param in point:
            counts = multinomial.rvs(self.n_draws, param, size=n_samples)
            samples.append(counts[0]) if n_samples == 1 else samples.append(counts)
        return samples[0] if len(point) == 1 else gs.stack(samples)


class MultinomialMetric(RiemannianMetric):
    """Class for the Fisher information metric on multinomial distributions.

    The Fisher information metric on the $n$-simplex of multinomial
    distributions parameters can be obtained as the pullback metric of the
    $n$-sphere using the componentwise square root.

    References
    ----------
    .. [K2003] R. E. Kass. The Geometry of Asymptotic Inference. Statistical
        Science, 4(3): 188 - 234, 1989.
    """

    def __init__(self, dim, n_draws):
        super().__init__(dim=dim, shape=(dim + 1,))
        self.n_draws = n_draws
        self.sphere_metric = HypersphereMetric(dim)

    def metric_matrix(self, base_point=None):
        """Compute the inner-product matrix.

        Compute the inner-product matrix of the Fisher information metric
        at the tangent space at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim + 1]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        if base_point is None:
            raise ValueError(
                "A base point must be given to compute the " "metric matrix"
            )
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        mat = self.n_draws * from_vector_to_diagonal_matrix(1 / base_point)
        return gs.squeeze(mat)

    @staticmethod
    def simplex_to_sphere(point):
        """Send point of the simplex to the sphere.

        The map takes the square root of each component.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point on the simplex.

        Returns
        -------
        point_sphere : array-like, shape=[..., dim + 1]
            Point on the sphere.
        """
        return point ** (1 / 2)

    @staticmethod
    def sphere_to_simplex(point):
        """Send point of the sphere to the simplex.

        The map squares each component.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point on the sphere.

        Returns
        -------
        point_simplex : array-like, shape=[..., dim + 1]
            Point on the simplex.
        """
        return point**2

    def tangent_simplex_to_sphere(self, tangent_vec, base_point):
        """Send tangent vector of the simplex to tangent space of sphere.

        This is the differential of the simplex_to_sphere map.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vec to the simplex at base point.
        base_point : array-like, shape=[..., dim + 1]
            Point of the simplex.

        Returns
        -------
        tangent_vec_sphere : array-like, shape=[..., dim + 1]
            Tangent vec to the sphere at the image of
            base point by simplex_to_sphere.
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        tangent_vec_sphere = gs.einsum(
            "...i,...i->...i", tangent_vec, 1 / (2 * self.simplex_to_sphere(base_point))
        )
        return gs.squeeze(tangent_vec_sphere)

    @staticmethod
    def tangent_sphere_to_simplex(tangent_vec, base_point):
        """Send tangent vector of the sphere to tangent space of simplex.

        This is the differential of the sphere_to_simplex map.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vec to the sphere at base point.
        base_point : array-like, shape=[..., dim + 1]
            Point of the sphere.

        Returns
        -------
        tangent_vec_simplex : array-like, shape=[..., dim + 1]
            Tangent vec to the simplex at the image of
            base point by sphere_to_simplex.
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        tangent_vec_simplex = gs.einsum("...i,...i->...i", tangent_vec, 2 * base_point)
        return gs.squeeze(tangent_vec_simplex)

    def exp(self, tangent_vec, base_point):
        """Compute the exponential map.

        Comute the exponential map associated to the Fisher information
        metric by pulling back the exponential map on the sphere by the
        simplex_to_sphere map.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., dim + 1]
            End point of the geodesic starting at base_point with
            initial velocity tangent_vec and stopping at time 1.
        """
        base_point_sphere = self.simplex_to_sphere(base_point)
        tangent_vec_sphere = self.tangent_simplex_to_sphere(tangent_vec, base_point)
        exp_sphere = self.sphere_metric.exp(tangent_vec_sphere, base_point_sphere)

        return self.sphere_to_simplex(exp_sphere)

    def log(self, point, base_point):
        """Compute the logarithm map.

        Compute logarithm map associated to the Fisher information
        metric by pulling back the exponential map on the sphere by
        the simplex_to_sphere map.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point.
        base_point : array-like, shape=[..., dim + 1]
            Base po int.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim + 1]
            Initial velocity of the geodesic starting at base_point and
            reaching point at time 1.
        """
        point_sphere = self.simplex_to_sphere(point)
        base_point_sphere = self.simplex_to_sphere(base_point)
        log_sphere = self.sphere_metric.log(point_sphere, base_point_sphere)

        return self.tangent_sphere_to_simplex(log_sphere, base_point_sphere)

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        """Generate parameterized function for the geodesic curve.

        Geodesic curve defined by either:

        - an initial point and an initial tangent vector,
        - an initial point and an end point.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim + 1]
            Point on the manifold, initial point of the geodesic.
        end_point : array-like, shape=[..., dim + 1]
            Point on the manifold, end point of the geodesic.
            Optional, default: None.
            If None, an initial tangent vector must be given.
        initial_tangent_vec : array-like, shape=[..., dim + 1],
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
        initial_point_sphere = self.simplex_to_sphere(initial_point)
        end_point_sphere = None
        vec_sphere = None
        if end_point is not None:
            end_point_sphere = self.simplex_to_sphere(end_point)
        if initial_tangent_vec is not None:
            vec_sphere = self.tangent_simplex_to_sphere(
                initial_tangent_vec, initial_point
            )
        geodesic_sphere = self.sphere_metric.geodesic(
            initial_point_sphere, end_point_sphere, vec_sphere
        )

        def path(t):
            """Generate parameterized function for geodesic curve.

            Parameters
            ----------
            t : array-like, shape=[n_times,]
                Times at which to compute points of the geodesics.

            Returns
            -------
            geodesic : array-like, shape=[..., n_times, dim + 1]
                Values of the geodesic at times t.
            """
            geod_sphere_at_t = geodesic_sphere(t)
            geod_at_t = self.sphere_to_simplex(geod_sphere_at_t)
            return gs.squeeze(geod_at_t)

        return path

    def sectional_curvature(self, tangent_vec_a, tangent_vec_b, base_point=None):
        r"""Compute the sectional curvature.

        In the literature sectional curvature is noted K.

        For two orthonormal tangent vectors :math:`x,y` at a base point,
        the sectional curvature is defined by :math:`K(x,y) = <R(x, y)x, y>`.
        For non-orthonormal vectors, it is
        :math:`K(x,y) = <R(x, y)x, y> / \\|x \wedge y\\|^2`.

        sectional_curvature(X, Y, P) = K(X,Y) where X, Y are tangent vectors
        at base point P.

        The information manifold of multinomial distributions has constant
        sectional curvature given by :math:`K = 2 \sqrt{n}`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., n, n]
            Point in the manifold.

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at `base_point`.

        Reference
        ---------
        [CF1992] Do Carmo, M. P., & Flaherty Francis, J. (1992).
        Riemannian geometry (Vol. 6). Boston: Birkhäuser.
        """
        sectional_curv = 2 * gs.sqrt(self.n_draws)
        if tangent_vec_a.ndim == 1 and tangent_vec_b.ndim == 1:
            if base_point is None or base_point.ndim == 1:
                return gs.array(sectional_curv)

        n_sec_curv = []
        if base_point is not None and base_point.ndim == 2:
            n_sec_curv.append(base_point.shape[0])
        if tangent_vec_a.ndim == 2:
            n_sec_curv.append(tangent_vec_a.shape[0])
        if tangent_vec_b.ndim == 2:
            n_sec_curv.append(tangent_vec_b.shape[0])
        n_sec_curv = gs.max(n_sec_curv)

        return gs.reshape(sectional_curv, (n_sec_curv,))
