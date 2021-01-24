"""Statistical Manifold of Dirichlet distributions with the Fisher metric."""

from scipy.integrate import odeint
from scipy.integrate import solve_bvp
from scipy.stats import dirichlet

import geomstats.backend as gs
import geomstats.errors
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric

N_STEPS = 100


class DirichletDistributions(EmbeddedManifold):
    """Class for the manifold of Dirichlet distributions.

    This is :math: Dirichlet = `(R_+^*)^dim`, the positive quadrant of the
    dim-dimensional Euclidean space.

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

        Check that point defines parameters for a Dirichlet distributions,
        i.e. belongs to the positive quadrant of the Euclidean space.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to be checked.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean indicating whether point represents a Dirichlet
            distribution.
        """
        point_dim = point.shape[-1]
        belongs = point_dim == self.dim
        belongs = gs.logical_and(
            belongs, gs.all(gs.greater(point, 0.), axis=-1))
        return belongs

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
        by point. This gives n_samples points in the simplex.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a Dirichlet distribution.
        n_samples : int
            Number of points to sample for each set of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from the Dirichlet distributions.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=2)
        samples = []
        for param in point:
            samples.append(gs.array(
                dirichlet.rvs(param, size=n_samples)))
        return samples[0] if len(point) == 1 else gs.stack(samples)

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
        geomstats.errors.check_belongs(point, self)

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
            pdf_at_x = []
            for param in point:
                pdf_at_x.append([
                    gs.array(dirichlet.pdf(pt, param)) for pt in x])
            pdf_at_x = gs.stack(pdf_at_x, axis=0)

            return pdf_at_x
        return pdf


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
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        n_points = base_point.shape[0]

        mat_ones = gs.ones((n_points, self.dim, self.dim))
        poly_sum = gs.polygamma(1, gs.sum(base_point, -1))
        mat_diag = from_vector_to_diagonal_matrix(
            gs.polygamma(1, base_point))

        mat = mat_diag - gs.einsum('i,ijk->ijk', poly_sum, mat_ones)
        return gs.squeeze(mat)

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
            c1 = 1 / gs.polygamma(1, param_k) / (
                1 / gs.polygamma(1, param_sum)
                - gs.sum(1 / gs.polygamma(1, base_point), -1))
            c2 = - c1 * gs.polygamma(2, param_sum) / gs.polygamma(1, param_sum)

            mat_ones = gs.ones((n_points, self.dim, self.dim))
            mat_diag = from_vector_to_diagonal_matrix(
                - gs.polygamma(2, base_point) / gs.polygamma(1, base_point))
            arrays = [gs.zeros((1, ind_k)),
                      gs.ones((1, 1)),
                      gs.zeros((1, self.dim - ind_k - 1))]
            vec_k = gs.tile(gs.hstack(arrays), (n_points, 1))
            val_k = gs.polygamma(2, param_k) / gs.polygamma(1, param_k)
            vec_k = gs.einsum('i,ij->ij', val_k, vec_k)
            mat_k = from_vector_to_diagonal_matrix(vec_k)

            mat = gs.einsum('i,ijk->ijk', c2, mat_ones)\
                - gs.einsum('i,ijk->ijk', c1, mat_diag) + mat_k

            return 1 / 2 * mat

        christoffels = []
        for ind_k in range(self.dim):
            christoffels.append(coefficients(ind_k))
        christoffels = gs.stack(christoffels, 1)

        return gs.squeeze(christoffels)

    def _geodesic_ivp(self, initial_point, initial_tangent_vec,
                      n_steps=N_STEPS):
        """Solve geodesic initial value problem.

        Compute the parameterized function for the geodesic starting at
        initial_point with initial velocity given by initial_tangent_vec.
        This is acheived by integrating the geodesic equation.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Initial point.

        initial_tangent_vec : array-like, shape=[..., dim]
            Tangent vector at initial point.

        Returns
        -------
        path : function
            Parameterized function for the geodesic curve starting at
            initial_point with velocity initial_tangent_vec.
        """
        initial_point = gs.to_ndarray(initial_point, to_ndim=2)
        initial_tangent_vec = gs.to_ndarray(initial_tangent_vec, to_ndim=2)

        n_initial_points = initial_point.shape[0]
        n_initial_tangent_vecs = initial_tangent_vec.shape[0]
        if n_initial_points > n_initial_tangent_vecs:
            raise ValueError('There cannot be more initial points than '
                             'initial tangent vectors.')
        if n_initial_tangent_vecs > n_initial_points:
            if n_initial_points > 1:
                raise ValueError('For several initial tangent vectors, '
                                 'specify either one or the same number of '
                                 'initial points.')
            initial_point = gs.tile(initial_point, (n_initial_tangent_vecs, 1))

        def ivp(state, _):
            """Reformat the initial value problem geodesic ODE."""
            position, velocity = state[:self.dim], state[self.dim:]
            eq = self.geodesic_equation(velocity=velocity, position=position)
            return gs.hstack(eq)

        def path(t):
            """Generate parameterized function for geodesic curve.

            Parameters
            ----------
            t : array-like, shape=[n_times,]
                Times at which to compute points of the geodesics.

            Returns
            -------
            geodesic : array-like, shape=[..., n_times, dim]
                Values of the geodesic at times t.
            """
            t = gs.to_ndarray(t, to_ndim=1)
            n_times = len(t)
            geod = []

            if n_times < n_steps:
                t_int = gs.linspace(0, 1, n_steps + 1)
                tangent_vecs = gs.einsum(
                    'i,...k->...ik', t, initial_tangent_vec)
                for point, vec in zip(initial_point, tangent_vecs):
                    point = gs.tile(point, (n_times, 1))
                    exp = []
                    for pt, vc in zip(point, vec):
                        initial_state = gs.hstack([pt, vc])
                        solution = odeint(
                            ivp, initial_state, t_int, (), rtol=1e-6)
                        exp.append(solution[-1, :self.dim])
                    exp = exp[0] if n_times == 1 else gs.stack(exp)
                    geod.append(exp)
            else:
                t_int = t
                for point, vec in zip(initial_point, initial_tangent_vec):
                    initial_state = gs.hstack([point, vec])
                    solution = odeint(
                        ivp, initial_state, t_int, (), rtol=1e-6)
                    geod.append(solution[:, :self.dim])

            return geod[0] if len(initial_point) == 1 else \
                gs.stack(geod)  # , axis=1)

        return path

    def exp(self, tangent_vec, base_point, n_steps=N_STEPS):
        """Compute the exponential map.

        Comute the exponential map associated to the Fisher information metric
        by solving the initial value problem associated to the geodesic
        ordinary differential equation (ODE) using the Christoffel symbols.

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
            End point of the geodesic starting at base_point with
            initial velocity tangent_vec and stopping at time 1.
        """
        stop_time = 1.
        geodesic = self._geodesic_ivp(base_point, tangent_vec, n_steps)
        exp = geodesic(stop_time)

        return exp

    def _geodesic_bvp(self, initial_point, end_point, n_steps=N_STEPS):
        """Solve geodesic boundary problem.

        Compute the parameterized function for the geodesic starting at
        initial_point and ending at end_point. This is acheived by integrating
        the geodesic equation.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Initial point.
        end_point : array-like, shape=[..., dim]
            End point.

        Returns
        -------
        path : function
            Parameterized function for the geodesic curve starting at
            initial_point and ending at end_point.
        """
        initial_point = gs.to_ndarray(initial_point, to_ndim=2)
        end_point = gs.to_ndarray(end_point, to_ndim=2)
        n_initial_points = initial_point.shape[0]
        n_end_points = end_point.shape[0]
        if n_initial_points > n_end_points:
            if n_end_points > 1:
                raise ValueError('For several initial points, specify either'
                                 'one or the same number of end points.')
            end_point = gs.tile(end_point, (n_initial_points, 1))
        elif n_end_points > n_initial_points:
            if n_initial_points > 1:
                raise ValueError('For several end points, specify either '
                                 'one or the same number of initial points.')
            initial_point = gs.tile(initial_point, (n_end_points, 1))

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
            """Boundary condition for the geodesic ODE."""
            return gs.hstack((state_0[:self.dim] - point_0,
                              state_1[:self.dim] - point_1))

        def path(t):
            """Generate parameterized function for geodesic curve.

            Parameters
            ----------
            t : array-like, shape=[n_times,]
                Times at which to compute points of the geodesics.

            Returns
            -------
            geodesic : array-like, shape=[..., n_times, dim]
                Values of the geodesic at times t.
            """
            t = gs.to_ndarray(t, to_ndim=1)
            geod = []

            def initialize(point_0, point_1):
                """Initialize the solution of the boundary value problem."""
                lin_init = gs.zeros([2 * self.dim, n_steps])
                lin_init[:self.dim, :] = gs.transpose(
                    gs.linspace(point_0, point_1, n_steps))
                lin_init[self.dim:, :-1] = n_steps * (
                    lin_init[:self.dim, 1:] - lin_init[:self.dim, :-1])
                lin_init[self.dim:, -1] = lin_init[self.dim:, -2]
                return lin_init

            t_int = gs.linspace(0., 1., n_steps)

            for ip, ep in zip(initial_point, end_point):
                geodesic_init = initialize(ip, ep)

                def bc(y0, y1, ip=ip, ep=ep):
                    return boundary_cond(y0, y1, ip, ep)

                solution = solve_bvp(bvp, bc, t_int, geodesic_init)
                solution_at_t = solution.sol(t)
                geodesic = solution_at_t[:self.dim, :]
                geod.append(gs.squeeze(gs.transpose(geodesic)))

            return geod[0] if len(initial_point) == 1 else gs.stack(geod)

        return path

    def log(self, point, base_point, n_steps=N_STEPS):
        """Compute the logarithm map.

        Compute logarithm map associated to the Fisher information metric by
        solving the boundary value problem associated to the geodesic ordinary
        differential equation (ODE) using the Christoffel symbols.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.
        base_point : array-like, shape=[..., dim]
            Base po int.
        n_steps : int
            Number of steps for integration.
            Optional, default: 100.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Initial velocity of the geodesic starting at base_point and
            reaching point at time 1.
        """
        t = gs.linspace(0., 1., n_steps)
        geodesic = self._geodesic_bvp(
            initial_point=base_point, end_point=point)
        geodesic_at_t = geodesic(t)
        log = n_steps * (geodesic_at_t[..., 1, :] - geodesic_at_t[..., 0, :])

        return gs.squeeze(gs.stack(log))

    def geodesic(self, initial_point, end_point=None,
                 initial_tangent_vec=None, n_steps=N_STEPS):
        """Generate parameterized function for the geodesic curve.

        Geodesic curve defined by either:
        - an initial point and an initial tangent vector,
        - an initial point and an end point.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Point on the manifold, initial point of the geodesic.
        end_point : array-like, shape=[..., dim], optional
            Point on the manifold, end point of the geodesic. If None,
            an initial tangent vector must be given.
        initial_tangent_vec : array-like, shape=[..., dim],
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
            raise ValueError('Specify an end point or an initial tangent '
                             'vector to define the geodesic.')
        if end_point is not None:
            if initial_tangent_vec is not None:
                raise ValueError('Cannot specify both an end point '
                                 'and an initial tangent vector.')
            path = self._geodesic_bvp(initial_point, end_point, n_steps)

        if initial_tangent_vec is not None:
            path = self._geodesic_ivp(
                initial_point, initial_tangent_vec, n_steps)

        return path
