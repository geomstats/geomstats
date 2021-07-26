"""Statistical Manifold of Dirichlet distributions with the Fisher metric."""

import logging
import math
import multiprocessing

from scipy.integrate import odeint
from scipy.integrate import solve_bvp
from scipy.stats import dirichlet

import geomstats.backend as gs
import geomstats.errors
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric

N_STEPS = 100
MAX_TIME = 300


class DirichletDistributions(OpenSet):
    """Class for the manifold of Dirichlet distributions.

    This is :math: Dirichlet = `(R_+^*)^dim`, the positive quadrant of the
    dim-dimensional Euclidean space.

    Attributes
    ----------
    dim : int
        Dimension of the manifold of Dirichlet distributions.
    """

    def __init__(self, dim):
        super(DirichletDistributions, self).__init__(dim=dim,
                                                     ambient_space=Euclidean(
                                                         dim=dim),
                                                     metric=DirichletMetric(
                                                         dim=dim))

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold of Dirichlet distributions.

        Check that point defines parameters for a Dirichlet distributions,
        i.e. belongs to the positive quadrant of the Euclidean space.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
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
        point_dim = point.shape[-1]
        belongs = point_dim == self.dim
        belongs = gs.logical_and(
            belongs, gs.all(point >= atol, axis=-1))
        return belongs

    def random_point(self, n_samples=1, bound=5.):
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

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the open set.

        The last coordinate is floored to `gs.atol` if it is negative.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in ambient space.
        atol : float
            Tolerance to evaluate positivity.

        Returns
        -------
        projected : array-like, shape=[..., dim]
            Projected point.
        """
        return gs.where(point < atol, atol, point)

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

        References
        ----------
        .. [LPP2021] A. Le Brigant, S. C. Preston, S. Puechmorel. Fisher-Rao
          geometry of Dirichlet Distributions. Differential Geometry
          and its Applications, 74, 101702, 2021.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        christoffels : array-like, shape=[..., dim, dim, dim]
            Christoffel symbols, with the contravariant index on
            the first dimension.
            :math: 'christoffels[..., i, j, k] = Gamma^i_{jk}'
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        n_points = base_point.shape[0]

        def coefficients(ind_k):
            """Christoffel symbols for contravariant index ind_k.
            """
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

    def jacobian_christoffels(self, base_point):
        """Compute the Jacobian of the Christoffel symbols.

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
            :math: 'jac[..., i, j, k, l] = dGamma^i_{jk} / dx_l'
        """
        n_dim = base_point.ndim
        param = gs.transpose(base_point)
        sum_param = gs.sum(param, 0)
        term_1 = 1 / gs.polygamma(1, param)
        term_2 = 1 / gs.polygamma(1, sum_param)
        term_3 = - gs.polygamma(2, param) / gs.polygamma(1, param)**2
        term_4 = - gs.polygamma(2, sum_param) / gs.polygamma(1, sum_param)**2
        term_5 = term_3 / term_1
        term_6 = term_4 / term_2
        term_7 = (gs.polygamma(2, param)**2 - gs.polygamma(1, param) *
                  gs.polygamma(3, param)) / gs.polygamma(1, param)**2
        term_8 = (gs.polygamma(2, sum_param)**2 - gs.polygamma(1, sum_param) *
                  gs.polygamma(3, sum_param)) / gs.polygamma(1, sum_param)**2
        term_9 = term_2 - gs.sum(term_1, 0)

        jac_1 = term_1 * term_8 / term_9
        jac_1_mat = gs.squeeze(
            gs.tile(jac_1, (self.dim, self.dim, self.dim, 1, 1)))
        jac_2 = - term_6 / term_9**2 * gs.einsum(
            'j...,i...->ji...', term_4 - term_3, term_1)
        jac_2_mat = gs.squeeze(
            gs.tile(jac_2, (self.dim, self.dim, 1, 1, 1)))
        jac_3 = term_3 * term_6 / term_9
        jac_3_mat = gs.transpose(
            from_vector_to_diagonal_matrix(gs.transpose(jac_3)))
        jac_3_mat = gs.squeeze(
            gs.tile(jac_3_mat, (self.dim, self.dim, 1, 1, 1)))
        jac_4 = 1 / term_9**2 * gs.einsum(
            'k...,j...,i...->kji...', term_5, term_4 - term_3, term_1)
        jac_4_mat = gs.transpose(
            from_vector_to_diagonal_matrix(gs.transpose(jac_4)))
        jac_5 = - gs.einsum('j...,i...->ji...', term_7, term_1) / term_9
        jac_5_mat = from_vector_to_diagonal_matrix(
            gs.transpose(jac_5))
        jac_5_mat = gs.transpose(from_vector_to_diagonal_matrix(
            jac_5_mat))
        jac_6 = - gs.einsum('k...,j...->kj...', term_5, term_3) / term_9
        jac_6_mat = gs.transpose(from_vector_to_diagonal_matrix(
            gs.transpose(jac_6)))
        jac_6_mat = gs.transpose(from_vector_to_diagonal_matrix(
            gs.transpose(jac_6_mat, [0, 1, 3, 2])), [0, 1, 3, 4, 2]) \
            if n_dim > 1 else from_vector_to_diagonal_matrix(
            jac_6_mat)
        jac_7 = - from_vector_to_diagonal_matrix(gs.transpose(term_7))
        jac_7_mat = from_vector_to_diagonal_matrix(jac_7)
        jac_7_mat = gs.transpose(
            from_vector_to_diagonal_matrix(jac_7_mat))

        jac = 1 / 2 * (
            jac_1_mat + jac_2_mat + jac_3_mat +
            jac_4_mat + jac_5_mat + jac_6_mat + jac_7_mat)

        return gs.transpose(jac, [3, 1, 0, 2]) if n_dim == 1 else \
            gs.transpose(jac, [4, 3, 1, 0, 2])

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
            state = gs.stack([position, velocity])
            vel, acc = self.geodesic_equation(state, _)
            eq = (vel, acc)
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
                            ivp, initial_state, t_int, ())
                        exp.append(solution[-1, :self.dim])
                    exp = exp[0] if n_times == 1 else gs.stack(exp)
                    geod.append(exp)
            else:
                t_int = t
                for point, vec in zip(initial_point, initial_tangent_vec):
                    initial_state = gs.hstack([point, vec])
                    solution = odeint(
                        ivp, initial_state, t_int, ())
                    geod.append(solution[:, :self.dim])

            return geod[0] if len(initial_point) == 1 else \
                gs.stack(geod)

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

    def _geodesic_bvp(self, initial_point, end_point, n_steps=N_STEPS,
                      jacobian=False, max_time=MAX_TIME):
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
        jacobian : boolean.
            If True, the explicit value of the jacobian is used to solve
            the geodesic boundary value problem.
            Optional, default: False.
        max_time : float.
            Maximum time in which the boundary value problem should be
            solved, in seconds. If it takes longer, the process is terminated.
            Optional, default: 300 seconds i.e. 5 minutes.

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
            state :  array-like, shape[2 * dim,]
                Vector of the state variables: position and speed.
            _ :  unused
                Any (time).
            """
            position, velocity = state[:self.dim].T, state[self.dim:].T
            state = gs.stack([position, velocity])
            vel, acc = self.geodesic_equation(state, _)
            eq = (vel, acc)
            return gs.transpose(gs.hstack(eq))

        def boundary_cond(
                state_0, state_1, point_0, point_1):
            """Boundary condition for the geodesic ODE."""
            return gs.hstack((state_0[:self.dim] - point_0,
                              state_1[:self.dim] - point_1))

        def jac(_, state):
            """Jacobian of bvp function.

            Parameters
            ----------
            state :  array-like, shape=[2*dim, ...]
                Vector of the state variables (position and speed)
            _ :  unused
                Any (time).

            Returns
            -------
            jac : array-like, shape=[dim, dim, ...]
            """
            n_dim = state.ndim
            n_times = state.shape[1] if n_dim > 1 else 1
            position, velocity = state[:self.dim], state[self.dim:]

            dgamma = self.jacobian_christoffels(gs.transpose(position))

            df_dposition = - gs.einsum(
                'j...,...ijkl,k...->il...', velocity, dgamma, velocity)

            gamma = self.christoffels(gs.transpose(position))
            df_dvelocity = - 2 * gs.einsum(
                '...ijk,k...->ij...', gamma, velocity)

            jac_nw = gs.zeros((self.dim, self.dim, state.shape[1])) \
                if n_dim > 1 else gs.zeros((self.dim, self.dim))
            jac_ne = gs.squeeze(gs.transpose(gs.tile(
                gs.eye(self.dim), (n_times, 1, 1))))
            jac_sw = df_dposition
            jac_se = df_dvelocity
            jac = gs.concatenate((
                gs.concatenate((jac_nw, jac_ne), axis=1),
                gs.concatenate((jac_sw, jac_se), axis=1)), axis=0)

            return jac

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
            fun_jac = jac if jacobian else None

            for ip, ep in zip(initial_point, end_point):

                def bc(y0, y1, ip=ip, ep=ep):
                    return boundary_cond(y0, y1, ip, ep)

                def process_function(return_dict, ip=ip, ep=ep):
                    solution = solve_bvp(
                        bvp, bc, t_int, initialize(ip, ep), fun_jac=fun_jac)
                    solution_at_t = solution.sol(t)
                    geodesic = solution_at_t[:self.dim, :]
                    geod.append(gs.squeeze(gs.transpose(geodesic)))
                    return_dict[0] = geod

                manager = multiprocessing.Manager()
                return_dict = manager.dict()
                process = multiprocessing.Process(
                    target=process_function, args=(return_dict,))
                process.start()

                process.join(max_time)
                if process.is_alive():
                    process.terminate()
                    logging.info('Maximum time of {} seconds reached. '
                                 'Process terminated. '
                                 'Result is inaccurate.'.format(max_time))
                    geod.append(math.nan * gs.zeros((n_steps, self.dim)))
                else:
                    geod = return_dict[0]

            return geod[0] if len(initial_point) == 1 else gs.stack(geod)

        return path

    def log(self, point, base_point, n_steps=N_STEPS, jacobian=False,
            max_time=MAX_TIME):
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
        jacobian : boolean.
            If True, the explicit value of the jacobian is used to solve
            the geodesic boundary value problem.
            Optional, default: False.
        max_time : float.
            Maximum time in which the boundary value problem should be
            solved, in seconds. If it takes longer, the process is terminated.
            Optional, default: 300 seconds i.e. 5 minutes.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Initial velocity of the geodesic starting at base_point and
            reaching point at time 1.
        """
        t = gs.linspace(0., 1., n_steps)
        geodesic = self._geodesic_bvp(
            initial_point=base_point, end_point=point, jacobian=jacobian,
            max_time=max_time)
        geodesic_at_t = geodesic(t)
        log = n_steps * (geodesic_at_t[..., 1, :] - geodesic_at_t[..., 0, :])

        return gs.squeeze(gs.stack(log))

    def geodesic(self, initial_point, end_point=None,
                 initial_tangent_vec=None, n_steps=N_STEPS,
                 jacobian=False, max_time=MAX_TIME):
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
        jacobian : boolean.
            If True, the explicit value of the jacobian is used to solve
            the geodesic boundary value problem.
            Optional, default: False.
        max_time : float.
            Maximum time in which the boundary value problem should be
            solved, in seconds. If it takes longer, the process is terminated.
            Optional, default: 300 seconds i.e. 5 minutes.

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
            path = self._geodesic_bvp(initial_point, end_point, n_steps,
                                      jacobian=jacobian, max_time=max_time)

        if initial_tangent_vec is not None:
            path = self._geodesic_ivp(
                initial_point, initial_tangent_vec, n_steps)

        return path
