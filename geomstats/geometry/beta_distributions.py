"""Statistical Manifold of beta distributions with the Fisher metric."""

from autograd.scipy.special import polygamma
from autograd.scipy.stats import beta
from autograd.scipy.integrate import odeint

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric

N_STEPS = 100


class BetaDistributions(EmbeddedManifold):
    """
    Class for the manifold of beta distributions
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
        point
        point_type

        Returns
        -------

        """
        point = gs.to_ndarray(point, to_ndim=2)
        n_points, point_dim = point.shape
        belongs = point_dim == self.dimension
        belongs = gs.to_ndarray(belongs, to_ndim=1)
        belongs = gs.tile(belongs, n_points)
        belongs = belongs * gs.greater(point, 0).all(axis=1)
        if n_points == 1:
            belongs = belongs[0]
        return belongs

    def random_uniform(self, n_samples=1, bound=10.0):
        """Sample parameters of beta distributions with the uniform
        distribution.

        Parameters
        ----------
        n_samples : int, optional
        bound: float, optional

        Returns
        -------
        samples : array-like, shape=[n_samples, 2]
        """
        return bound * gs.random.rand(n_samples, 2)

    def maximum_likelihood_fit(self, data, loc=0, scale=0):
        parameters = []
        for sample in data:
            a, b, _, _ = beta.fit(sample, floc=loc, fscale=scale)
            parameters.append(gs.array([a, b]))
        return gs.stack(parameters)


class BetaMetric(RiemannianMetric):

    def __init__(self):
        super(RiemannianMetric, self).__init__(dimension=2)

    @staticmethod
    def detg(param_a, param_b):
        detg = polygamma(1, param_a) * polygamma(1, param_b) - \
            polygamma(1, param_a + param_b) * (polygamma(1, param_a) +
                                               polygamma(1, param_b))
        return detg

    def inner_product_matrix(self, base_point=None):
        """Compute inner product matrix at the tangent space at base point.

        Parameters
        ----------
        base_point : array-like, shape=[n_samples, dimension]

        """
        assert ~ (base_point is None), 'The metric depends on the base point'
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        matrices = []
        for point in base_point:
            param_a, param_b = point
            g0 = gs.array([polygamma(1, param_a) - polygamma(1, param_a +
                                                             param_b),
                           - polygamma(1, param_a + param_b)])
            g1 = gs.array([- polygamma(1, param_a + param_b),
                           polygamma(1, param_b) -
                           polygamma(1, param_a + param_b)])
            matrices.append(gs.stack([g0, g1]))
        return gs.stack(matrices)

    @staticmethod
    def christoffels(base_point):
        """Compute Christoffel symbols.

        Compute the Christoffel symbols of the Fisher metric on Beta
        distributions.
        """

        def coefficients(param_a, param_b):
            detg = self.detg(param_a, param_b)
            c1 = (polygamma(2, param_a) * polygamma(1, param_b) -
                  polygamma(2, param_a) * polygamma(1, param_a + param_b) -
                  polygamma(1, param_b) * polygamma(2, param_a + param_b)) / \
                 (2 * detg)
            c2 = - polygamma(1, param_b) * polygamma(2, param_a + param_b) / (
                    2 * detg)
            c3 = (polygamma(2, param_b) * polygamma(1, param_a + param_b) -
                  polygamma(1, param_b) * polygamma(2, param_a + param_b)) / (
                    2 * detg)
            return c1, c2, c3

        assert ~ (base_point is None), 'The Christoffels require a base point'
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        param_a, param_b = base_point[:, 0], base_point[:, 1]
        c1, c2, c3 = coefficients(param_a, param_b)
        c4, c5, c6 = coefficients(param_b, param_a)
        christoffel = []
        for d1, d2, d3, d4, d5, d6 in zip(c1, c2, c3, c4, c5, c6):
            gamma_0 = gs.array([[d1, d2], [d2, d3]])
            gamma_1 = gs.array([[d6, d5], [d5, d4]])
            christoffel.append(gs.stack([gamma_0, gamma_1]))
        return christoffel[0] if len(base_point) == 1 else gs.stack(christoffel)

    def exp(self, tangent_vec, base_point, n_steps=N_STEPS):
        """
        Exponential map.
        """

        def func_ivp(state, time):
            """Reformat the differential equations.

            Parameters
            ----------
            time
            state
            """
            point, velocity = state[:2], state[2:]
            eq = self.geodesic_equation(tangent_vec=velocity, base_point=point)
            return gs.hstack([velocity, eq])

        times = gs.linspace(0, 1, n_steps + 1)
        y0 = gs.hstack([base_point, tangent_vec])
        geodesic = odeint(func_ivp, y0, times, tuple(), rtol=1e-6)
        return geodesic[-1, :2]

    def log(self, point, base_point, n_steps=N_STEPS):
        a0, b0 = base_point
        a1, b1 = point

        stop_time = 1.
        t = [stop_time * float(i) / (n_steps - 1) for i in range(n_steps)]
        geodesic_init = gs.zeros([2 * dim, n_points])
        geodesic_init[0, :] = gs.linspace(a0, a1, n_steps)
        geodesic_init[1, :] = gs.linspace(b0, b1, n_steps)
        geodesic_init[2, :-1] = n_steps * (geodesic_init[0, 1:] -
                                            geodesic_init[0, :-1])
        geodesic_init[3, :-1] = n_steps * (geodesic_init[1, 1:] -
                                            geodesic_init[1, :-1])
        geodesic_init[2, -1] = geodesic_init[2, -2]
        geodesic_init[3, -1] = geodesic_init[3, -2]

<<<<<<< HEAD
        def func_bvp(time, state):
            """Reformat the differential equation.

            Parameters
            ----------
                y :  vector of the state variables: y = [a,b,u,v]
                x :  time
            """
            point, velocity = state[:2], state[2:]
            eq = self.geodesic_equation(tangent_vec=velocity, base_point=point)
            return gs.hstack([velocity, eq])

        def boundary_cond(y0, y1):
=======
        def boundary_cond(y0, y1):

>>>>>>> fba2b7e31f406416fa953bb20ef8bd609a70d3b5
            bc = gs.array([y0[0] - a0,
                           y0[1] - b0,
                           y1[0] - a1,
                           y1[1] - b1])
            return bc

        solution = solve_bvp(func_bvp, boundary_cond, t, geodesic_init)

        geodesic = solution.sol(t)
        geodesic = geodesic[:2, :]
