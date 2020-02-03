"""Statistical Manifold of beta distributions with the Fisher metric."""

from autograd.scipy.special import polygamma
from scipy.integrate import odeint
from scipy.integrate import solve_bvp
from scipy.stats import beta

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric

N_STEPS = 100
EPSILON = 1e-6


class BetaDistributions(EmbeddedManifold):
    r"""Class for the manifold of beta distributions.

    This is :math: Beta = `R_+^* \times R_+^*`.
    """

    def __init__(self):
        super(BetaDistributions, self).__init__(
            dimension=2,
            embedding_manifold=Euclidean(dimension=2))

    def belongs(self, point, point_type=None):
        """Evaluate if a point belongs to the manifold of beta distributions.

        The statistical manifold of beta distributions is the upper right
        quadrant of the euclidean 2-plane.

        Parameters
        ----------
        point : array-like, shape=[n_samples, 2]
            the point of which to check whether it belongs to Beta

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
            array of booleans indicating whether point belongs the Beta
        """
        point = gs.to_ndarray(point, to_ndim=2)
        n_points, point_dim = point.shape
        belongs = point_dim == self.dimension
        belongs = gs.to_ndarray(belongs, to_ndim=1)
        belongs = gs.tile(belongs, n_points)
        belongs = belongs * gs.greater(point, 0).all(axis=1)
        return belongs[0] if n_points == 1 else belongs

    @staticmethod
    def random_uniform(n_samples=1, bound=10.0):
        """Sample parameters of beta distributions.

        The uniform distribution on [0, bound]^2 is used.

        Parameters
        ----------
        n_samples : int, optional
        bound : float, optional
            scalar to scale samples

        Returns
        -------
        samples : array-like, shape=[n_samples, 2]
        """
        return bound * gs.random.rand(n_samples, 2)

    def sample(self, point, n_samples=1):
        """Sample from the beta distribution.

        Sample from the beta distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape [n_points, 2]
        n_samples : int
            the number of points to sample with each pair of parameter in
            point

        Returns
        -------
        samples : array-like, shape=[n_points, n_samples]
        """
        point = gs.to_ndarray(point, to_ndim=2)
        assert self.belongs(point).all()
        samples = []
        for param_a, param_b in point:
            samples.append(beta.rvs(param_a, param_b, size=n_samples))
        return samples[0] if len(point) == 1 else gs.stack(samples)

    def maximum_likelihood_fit(self, data, loc=0, scale=1):
        """Estimate parameters from samples.

        This a wrapper around scipy's maximum likelihood estimator to
        estimate the parameters of a beta distribution from samples.

        Parameters
        ----------
        data : array-like, shape=[n_distributions, n_samples]
            the data to estimate parameters from. Arrays of
            different length may be passed.
        loc : float, optional
            the location parameter of the distribution to estimate parameters
            from. It is kept fixed during optimization
            default: 0
        scale : float, optional
            the scale parameter of the distribution to estimate parameters
            from. It is kept fixed during optimization
            default: 1
        Returns
        -------
        parameter : array-like, shape=[n_samples, 2]
        """
        data = gs.to_ndarray(
            gs.where(data == 1., 1 - EPSILON, data), to_ndim=2)
        parameters = []
        for sample in data:
            param_a, param_b, _, _ = beta.fit(sample, floc=loc, fscale=scale)
            parameters.append(gs.array([param_a, param_b]))
        return parameters[0] if len(data) == 1 else gs.stack(parameters)


class BetaMetric(RiemannianMetric):
    """Class for the Fisher information metric on beta distributions."""

    def __init__(self):
        super(RiemannianMetric, self).__init__(dimension=2)

    @staticmethod
    def metric_det(param_a, param_b):
        """Compute the determinant of the metric.

        Parameters
        ----------
        param_a : array-like, shape=[n_samples,]
        param_b : array-like, shape=[n_samples,]

        Returns
        -------
        metric_det : array-like, shape=[n_samples,]
        """
        metric_det = polygamma(1, param_a) * polygamma(1, param_b) - \
            polygamma(1, param_a + param_b) * (polygamma(1, param_a) +
                                               polygamma(1, param_b))
        return metric_det

    def inner_product_matrix(self, base_point=None):
        """Compute inner product matrix at the tangent space at base point.

        Parameters
        ----------
        base_point : array-like, shape=[n_samples, 2]

        Returns
        -------
        base_point : array-like, shape=[n_samples, 2, 2]
        """
        assert base_point is not None, 'The metric depends on the base point'
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        matrices = []
        for point in base_point:
            param_a, param_b = point
            g0 = gs.array(
                [polygamma(1, param_a) - polygamma(1, param_a + param_b),
                 - polygamma(1, param_a + param_b)])
            g1 = gs.array(
                [- polygamma(1, param_a + param_b), polygamma(1, param_b)
                 - polygamma(1, param_a + param_b)])
            matrices.append(gs.stack([g0, g1]))
        return gs.stack(matrices)

    def christoffels(self, base_point):
        """Compute the Christoffel symbols.

        Compute the Christoffel symbols of the Fisher information metric on
        Beta.

        Parameters
        ----------
        base_point : array-like, shape=[n_samples, 2]

        Returns
        -------
        christoffels : array-like, shape=[n_samples, 2, 2, 2]
        """

        def coefficients(param_a, param_b):
            metric_det = self.metric_det(param_a, param_b)
            c1 = (polygamma(2, param_a) * polygamma(1, param_b) -
                  polygamma(2, param_a) * polygamma(1, param_a + param_b) -
                  polygamma(1, param_b) * polygamma(2, param_a + param_b)) / (
                2 * metric_det)
            c2 = - polygamma(1, param_b) * polygamma(2, param_a + param_b) / (
                2 * metric_det)
            c3 = (polygamma(2, param_b) * polygamma(1, param_a + param_b) -
                  polygamma(1, param_b) * polygamma(2, param_a + param_b)) / (
                2 * metric_det)
            return c1, c2, c3

        assert base_point is not None, 'The Christoffels require a base point'
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        param_a, param_b = base_point[:, 0], base_point[:, 1]
        c1, c2, c3 = coefficients(param_a, param_b)
        c4, c5, c6 = coefficients(param_b, param_a)
        christoffel = []
        for d1, d2, d3, d4, d5, d6 in zip(c1, c2, c3, c4, c5, c6):
            gamma_0 = gs.array([[d1, d2], [d2, d3]])
            gamma_1 = gs.array([[d6, d5], [d5, d4]])
            christoffel.append(gs.stack([gamma_0, gamma_1]))
        if len(base_point) == 1:
            return christoffel[0]
        else:
            return gs.stack(christoffel)

    def exp(self, tangent_vec, base_point, n_steps=N_STEPS):
        """Exponential map associated to the Fisher information metric.

        Exponential map at base_point of tangent_vec computed by integration
        of the geodesic equation (initial value problem), using the
        christoffel symbols.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, dimension]
        base_point : array-like, shape=[n_samples, dimension]
        n_steps : int

        Returns
        -------
        exp : array-like, shape=[n_samples, dimension]
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)

        def ivp(state, time):
            """Reformat the initial value problem geodesic ODE."""
            position, velocity = state[:2], state[2:]
            eq = self.geodesic_equation(velocity=velocity, position=position)
            return gs.hstack([velocity, eq])

        times = gs.linspace(0, 1, n_steps + 1)
        exp = []
        for point, vec in zip(base_point, tangent_vec):
            initial_state = gs.hstack([point, vec])
            geodesic = odeint(
                ivp, initial_state, times, tuple(), rtol=1e-6)
            exp.append(geodesic[-1, :2])
        return exp[0] if len(base_point) == 1 else gs.stack(exp)

    def log(self, point, base_point, n_steps=N_STEPS):
        """Compute logarithm map associated to the Fisher information metric.

        Solve the boundary value problem associated to the geodesic ordinary
        differential equation (ODE) using the Christoffel symbols.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
        base_point : array-like, shape=[n_samples, dimension]
        n_steps : int

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, dimension]
            the initial velocity of the geodesic starting at base_point and
            reaching point at time 1
        """
        stop_time = 1.
        t = gs.linspace(0, stop_time, n_steps)
        point = gs.to_ndarray(point, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        def initialize(end_point, start_point):
            a0, b0 = start_point
            a1, b1 = end_point
            lin_init = gs.zeros([2 * self.dimension, n_steps])
            lin_init[0, :] = gs.linspace(a0, a1, n_steps)
            lin_init[1, :] = gs.linspace(b0, b1, n_steps)
            lin_init[2, :-1] = (lin_init[0, 1:] - lin_init[0, :-1]) * n_steps
            lin_init[3, :-1] = (lin_init[1, 1:] - lin_init[1, :-1]) * n_steps
            lin_init[2, -1] = lin_init[2, -2]
            lin_init[3, -1] = lin_init[3, -2]
            return lin_init

        def bvp(time, state):
            """Reformat the boundary value problem geodesic ODE.

            Parameters
            ----------
                state :  vector of the state variables: y = [a,b,u,v]
                time :  time
            """
            position, velocity = state[:2].T, state[2:].T
            eq = self.geodesic_equation(
                velocity=velocity, position=position)
            return gs.vstack((velocity.T, eq.T))

        def boundary_cond(
                state_a, state_b, point_0_a, point_0_b, point_1_a, point_1_b):
            return gs.array(
                [state_a[0] - point_0_a,
                 state_a[1] - point_0_b,
                 state_b[0] - point_1_a,
                 state_b[1] - point_1_b])

        log = []
        for bp, pt in zip(base_point, point):
            geodesic_init = initialize(pt, bp)
            base_point_a, base_point_b = bp
            point_a, point_b = pt

            def bc(y0, y1):
                return boundary_cond(
                    y0, y1, base_point_a, base_point_b, point_a, point_b)

            solution = solve_bvp(bvp, bc, t, geodesic_init)
            geodesic = solution.sol(t)
            geodesic = geodesic[:2, :]
            log.append(n_steps * (geodesic[:, 1] - geodesic[:, 0]))

        return log[0] if len(base_point) == 1 else gs.stack(log)
