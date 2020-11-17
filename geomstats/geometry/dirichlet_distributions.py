"""Statistical Manifold of Dirichlet distributions with the Fisher metric."""

from numpy import diag
from scipy.integrate import odeint
from scipy.integrate import solve_bvp
from scipy.stats import dirichlet

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric

N_STEPS = 100
EPSILON = 1e-6


class DirichletDistributions(EmbeddedManifold):
    """Class for the manifold of beta distributions.

    This is :math: Dirichlet = `(R_+^*)^n`.

    Attributes
    ----------
    dim : int
        Dimension of the manifold of Dirichlet distributions.
    embedding_manifold : Manifold
        Embedding manifold.
    """

    def __init__(self, dim):
        super(DirichletDistributions, self).__init__(
            dim=dim,
            embedding_manifold=Euclidean(dim=dim))
        self.metric = DirichletMetric(dim=dim)

    def belongs(self, point):
        """Evaluate if a point belongs to the manifold of Dirichlet distributions.

        The statistical manifold of Dirichlet distributions is the upper right
        quadrant of the Euclidean space.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to be checked.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean indicating whether point represents a beta distribution.
        """
        point_dim = point.shape[-1]
        belongs = point_dim == self.dim
        belongs = gs.logical_and(
            belongs, gs.all(gs.greater(point, 0.), axis=-1))
        return belongs

    # @staticmethod
    def random_uniform(self, n_samples=1, bound=5.):
        """Sample parameters of Dirichlet distributions.

        The uniform distribution on [0, bound]^dim is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of the square where the Dirichlet parameters are sampled.
            Optional, default: 5.

        Returns
        -------
        samples : array-like, shape=[..., dim]
            Sample of points representing Dirichlet distributions.
        """
        size = (self.dim,) if n_samples == 1 else (n_samples, self.dim)
        return bound * gs.random.rand(*size)

    def sample(self, point, n_samples=1):
        """Sample from the Dirichlet distribution.

        Sample from the Dirichlet distribution with parameters provided
        by point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a Dirichlet distribution.
        n_samples : int
            Number of points to sample with each pair of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from beta distributions.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=2)
        samples = []
        for i in range(n_samples):
            samples.append(gs.array(
                dirichlet.rvs(point[i, :], size=n_samples)))
        return samples[0] if len(point) == 1 else gs.stack(samples)


class DirichletMetric(RiemannianMetric):
    """Class for the Fisher information metric on Dirichlet distributions."""

    def __init__(self, dim):
        super(DirichletMetric, self).__init__(dim=dim)

    def metric_matrix(self, base_point=None):
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
        if base_point is None:
            raise ValueError('A base point must be given to compute the '
                             'metric matrix')
        ones_mat = gs.ones((self.dim, self.dim))
        sum_param = gs.sum(base_point, -1)
        mat = diag(gs.polygamma(1, base_point)) \
            - gs.polygamma(1, sum_param) * ones_mat
        return mat

    def christoffels(self, base_point):
        """Compute the Christoffel symbols.

        Compute the Christoffel symbols of the Fisher information metric.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        christoffels : array-like, shape=[..., dim, dim, dim]
            Christoffel symbols.
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        n_points = base_point.shape[0]

        def coefficients(ind_k):
            param_k = base_point[..., ind_k]
            param_sum = gs.sum(base_point, -1)
            C1 = 1 / gs.polygamma(1, param_k) / (
                1 / gs.polygamma(1, param_sum)
                - gs.sum(1 / gs.polygamma(1, base_point), -1))
            C2 = - C1 * gs.polygamma(2, param_sum) / gs.polygamma(1, param_sum)

            mat_ones = gs.ones((n_points, self.dim, self.dim))
            mat_diag = gs.zeros((n_points, self.dim, self.dim))
            diag = gs.einsum('ijj->ij', mat_diag)
            diag[:] = - gs.polygamma(2, base_point) \
                / gs.polygamma(1, base_point)

            mat_k = gs.einsum('i,ijk->ijk', C2, mat_ones)\
                - gs.einsum('i,ijk->ijk', C1, mat_diag)
            mat_k[:, ind_k, ind_k] += gs.polygamma(2, param_k)\
                / gs.polygamma(1, param_k)
            return 1 / 2 * mat_k

        christoffels = gs.zeros((n_points, self.dim, self.dim, self.dim))
        for ind_k in range(self.dim):
            christoffels[:, ind_k, :, :] = coefficients(ind_k)

        return gs.squeeze(christoffels)

    def exp(self, tangent_vec, base_point, n_steps=N_STEPS):
        """Exponential map associated to the Fisher information metric.

        Exponential map at base_point of tangent_vec computed by integration
        of the geodesic equation (initial value problem), using the
        christoffel symbols.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Base point.
        n_steps : int
            Number of steps for integration.
            Optional, default: 100.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Riemannian exponential.
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)

        n_base_points = base_point.shape[0]
        n_tangent_vecs = tangent_vec.shape[0]
        if n_base_points > n_tangent_vecs:
            raise ValueError('There cannot be more base points than tangent '
                             'vectors.')
        if n_tangent_vecs > n_base_points:
            if n_base_points > 1:
                raise ValueError('For several tangent vectors, specify '
                                 'either one or the same number of base '
                                 'points.')
            base_point = gs.tile(base_point, (n_tangent_vecs, 1))

        def ivp(state, _):
            """Reformat the initial value problem geodesic ODE."""
            position, velocity = state[:self.dim], state[self.dim:]
            eq = self.geodesic_equation(velocity=velocity, position=position)
            return gs.hstack(eq)

        times = gs.linspace(0, 1, n_steps + 1)
        exp = []
        for point, vec in zip(base_point, tangent_vec):
            initial_state = gs.hstack([point, vec])
            geodesic = odeint(
                ivp, initial_state, times, (), rtol=1e-6)
            exp.append(geodesic[-1, :self.dim])
        return exp[0] if len(base_point) == 1 else gs.stack(exp)

    def log(self, point, base_point, n_steps=N_STEPS):
        """Compute logarithm map associated to the Fisher information metric.

        Solve the boundary value problem associated to the geodesic ordinary
        differential equation (ODE) using the Christoffel symbols.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.
        base_point : array-like, shape=[..., dim]
            Base point.
        n_steps : int
            Number of steps for integration.
            Optional, default: 100.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Initial velocity of the geodesic starting at base_point and
            reaching point at time 1.
        """
        stop_time = 1.
        t = gs.linspace(0, stop_time, n_steps)
        point = gs.to_ndarray(point, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        n_points = point.shape[0]
        n_base_points = base_point.shape[0]
        if n_base_points > n_points:
            if n_points > 1:
                raise ValueError('For several base points, specify either '
                                 'one or the same number of points.')
            point = gs.tile(point, (n_base_points, 1))
        elif n_points > n_base_points:
            if n_base_points > 1:
                raise ValueError('For several points, specify either '
                                 'one or the same number of base points.')
            base_point = gs.tile(base_point, (n_points, 1))

        def initialize(end_point, start_point):
            lin_init = gs.zeros([2 * self.dim, n_steps])
            lin_init[:self.dim, :] = gs.transpose(
                gs.linspace(start_point, end_point, n_steps))
            lin_init[self.dim:, :-1] = (
                lin_init[:self.dim, 1:] - lin_init[:self.dim, :-1]) * n_steps
            lin_init[self.dim:, -1] = lin_init[self.dim:, -2]
            return lin_init

        def bvp(_, state):
            """Reformat the boundary value problem geodesic ODE.

            Parameters
            ----------
            state :  array-like, shape[4,]
                Vector of the state variables: y = [a,b,u,v]
            _ :  unused
                Any (time).
            """
            position, velocity = state[:self.dim].T, state[self.dim:].T
            eq = self.geodesic_equation(
                velocity=velocity, position=position)
            return gs.transpose(gs.hstack(eq))

        def boundary_cond(
                state_0, state_1, point_0, point_1):
            return gs.hstack((state_0[:self.dim] - point_0,
                              state_1[:self.dim] - point_1))

        log = []
        for bp, pt in zip(base_point, point):
            geodesic_init = initialize(pt, bp)

            def bc(y0, y1, bp=bp, pt=pt):
                return boundary_cond(y0, y1, bp, pt)

            solution = solve_bvp(bvp, bc, t, geodesic_init)
            geodesic = solution.sol(t)
            geodesic = geodesic[:self.dim, :]
            log.append(n_steps * (geodesic[:, 1] - geodesic[:, 0]))

        return log[0] if len(base_point) == 1 else gs.stack(log)
