"""Statistical Manifold of Dirichlet distributions with the Fisher metric.

Lead author: Alice Le Brigant.
"""

import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import dirichlet

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.base import VectorSpaceOpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyMultivariateRandomVariable,
)
from geomstats.numerics.bvp import ScipySolveBVP
from geomstats.numerics.geodesic import ExpODESolver, LogODESolver
from geomstats.numerics.ivp import ScipySolveIVP
from geomstats.vectorization import repeat_out


class DirichletDistributions(InformationManifoldMixin, VectorSpaceOpenSet):
    """Class for the manifold of Dirichlet distributions.

    This is Dirichlet = :math:`(R_+^*)^dim`, the positive quadrant of the
    dim-dimensional Euclidean space.

    Attributes
    ----------
    dim : int
        Dimension of the manifold of Dirichlet distributions.
    """

    def __init__(self, dim, equip=True):
        super().__init__(
            dim=dim,
            support_shape=(dim,),
            embedding_space=Euclidean(dim=dim, equip=False),
            equip=equip,
        )
        self._scp_rv = DirichletRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return DirichletMetric

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
        belongs = point.shape[-1] == self.dim
        if not belongs:
            return gs.zeros(point.shape[:-1], dtype=bool)

        return gs.all(point >= -atol, axis=-1)

    def random_point(self, n_samples=1, bound=5.0):
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
        samples : array-like, shape=[..., n_samples, dim]
            Sample from the Dirichlet distributions.
        """
        return self._scp_rv.rvs(point, n_samples)

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
        return lambda x: self._scp_rv.pdf(x, point=point)


class DirichletMetric(RiemannianMetric):
    """Class for the Fisher information metric on Dirichlet distributions."""

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
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        n_points = base_point.shape[0]

        mat_ones = gs.ones((n_points, self._space.dim, self._space.dim))
        poly_sum = gs.polygamma(1, gs.sum(base_point, -1))
        mat_diag = from_vector_to_diagonal_matrix(gs.polygamma(1, base_point))

        mat = mat_diag - gs.einsum("i,ijk->ijk", poly_sum, mat_ones)
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
            Christoffel symbols, with the contravariant index on
            the first dimension.
            :math:`christoffels[..., i, j, k] = Gamma^i_{jk}`

        References
        ----------
        .. [LPP2021] A. Le Brigant, S. C. Preston, S. Puechmorel. Fisher-Rao
            geometry of Dirichlet Distributions. Differential Geometry
            and its Applications, 74, 101702, 2021.
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        n_points = base_point.shape[0]

        def coefficients(ind_k):
            """Christoffel symbols for contravariant index ind_k."""
            param_k = base_point[..., ind_k]
            param_sum = gs.sum(base_point, -1)
            c1 = (
                1
                / gs.polygamma(1, param_k)
                / (
                    1 / gs.polygamma(1, param_sum)
                    - gs.sum(1 / gs.polygamma(1, base_point), -1)
                )
            )
            c2 = -c1 * gs.polygamma(2, param_sum) / gs.polygamma(1, param_sum)

            mat_ones = gs.ones((n_points, self._space.dim, self._space.dim))
            mat_diag = from_vector_to_diagonal_matrix(
                -gs.polygamma(2, base_point) / gs.polygamma(1, base_point)
            )
            arrays = [
                gs.zeros((1, ind_k)),
                gs.ones((1, 1)),
                gs.zeros((1, self._space.dim - ind_k - 1)),
            ]
            vec_k = gs.tile(gs.hstack(arrays), (n_points, 1))
            val_k = gs.polygamma(2, param_k) / gs.polygamma(1, param_k)
            vec_k = gs.einsum("i,ij->ij", val_k, vec_k)
            mat_k = from_vector_to_diagonal_matrix(vec_k)

            mat = (
                gs.einsum("i,ijk->ijk", c2, mat_ones)
                - gs.einsum("i,ijk->ijk", c1, mat_diag)
                + mat_k
            )

            return 1 / 2 * mat

        christoffels = []
        for ind_k in range(self._space.dim):
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
            :math:`jac[..., i, j, k, l] = dGamma^i_{jk} / dx_l`
        """
        dim = self._space.dim
        n_dim = base_point.ndim
        param = gs.transpose(base_point)
        sum_param = gs.sum(param, 0)
        term_1 = 1 / gs.polygamma(1, param)
        term_2 = 1 / gs.polygamma(1, sum_param)
        term_3 = -gs.polygamma(2, param) / gs.polygamma(1, param) ** 2
        term_4 = -gs.polygamma(2, sum_param) / gs.polygamma(1, sum_param) ** 2
        term_5 = term_3 / term_1
        term_6 = term_4 / term_2
        term_7 = (
            gs.polygamma(2, param) ** 2
            - gs.polygamma(1, param) * gs.polygamma(3, param)
        ) / gs.polygamma(1, param) ** 2
        term_8 = (
            gs.polygamma(2, sum_param) ** 2
            - gs.polygamma(1, sum_param) * gs.polygamma(3, sum_param)
        ) / gs.polygamma(1, sum_param) ** 2
        term_9 = term_2 - gs.sum(term_1, 0)

        jac_1 = term_1 * term_8 / term_9
        jac_1_mat = gs.squeeze(gs.tile(jac_1, (dim, dim, dim, 1, 1)))
        jac_2 = (
            -term_6 / term_9**2 * gs.einsum("j...,i...->ji...", term_4 - term_3, term_1)
        )
        jac_2_mat = gs.squeeze(gs.tile(jac_2, (dim, dim, 1, 1, 1)))
        jac_3 = term_3 * term_6 / term_9
        jac_3_mat = gs.transpose(from_vector_to_diagonal_matrix(gs.transpose(jac_3)))
        jac_3_mat = gs.squeeze(gs.tile(jac_3_mat, (dim, dim, 1, 1, 1)))
        jac_4 = (
            1
            / term_9**2
            * gs.einsum("k...,j...,i...->kji...", term_5, term_4 - term_3, term_1)
        )
        jac_4_mat = gs.transpose(from_vector_to_diagonal_matrix(gs.transpose(jac_4)))
        jac_5 = -gs.einsum("j...,i...->ji...", term_7, term_1) / term_9
        jac_5_mat = from_vector_to_diagonal_matrix(gs.transpose(jac_5))
        jac_5_mat = gs.transpose(from_vector_to_diagonal_matrix(jac_5_mat))
        jac_6 = -gs.einsum("k...,j...->kj...", term_5, term_3) / term_9
        jac_6_mat = gs.transpose(from_vector_to_diagonal_matrix(gs.transpose(jac_6)))
        jac_6_mat = (
            gs.transpose(
                from_vector_to_diagonal_matrix(gs.transpose(jac_6_mat, [0, 1, 3, 2])),
                [0, 1, 3, 4, 2],
            )
            if n_dim > 1
            else from_vector_to_diagonal_matrix(jac_6_mat)
        )
        jac_7 = -from_vector_to_diagonal_matrix(gs.transpose(term_7))
        jac_7_mat = from_vector_to_diagonal_matrix(jac_7)
        jac_7_mat = gs.transpose(from_vector_to_diagonal_matrix(jac_7_mat))

        jac = (
            1
            / 2
            * (
                jac_1_mat
                + jac_2_mat
                + jac_3_mat
                + jac_4_mat
                + jac_5_mat
                + jac_6_mat
                + jac_7_mat
            )
        )

        return (
            gs.transpose(jac, [3, 1, 0, 2])
            if n_dim == 1
            else gs.transpose(jac, [4, 3, 1, 0, 2])
        )

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base point onto
        its image.
        In the case of the hyperbolic space, it does not depend on the base point and
        is infinite everywhere, because of the negative curvature.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)

    def _approx_geodesic_bvp(
        self,
        initial_point,
        end_point,
        degree=5,
        method="BFGS",
        n_times=200,
        jac_on=True,
    ):
        """Solve approximation of the geodesic boundary value problem.

        The space of solutions is restricted to curves whose coordinates are
        polynomial functions of time. The boundary value problem is solved by
        minimizing the energy among all such curves starting from initial_point
        and ending at end_point, i.e. curves t -> (x_1(t),...,x_n(t)) where x_i
        are polynomial functions of time t, such that (x_1(0),..., x_n(0)) is
        initial_point and (x_1(1),..., x_n(1)) is end_point. The parameterized
        curve is computed at n_times discrete times.

        Parameters
        ----------
        initial_point : array-like, shape=(dim,)
            Starting point of the geodesic.
        end_point : array-like, shape=(dim,)
            End point of the geodesic.
        degree : int
            Degree of the coordinates' polynomial functions of time.
        method : str
            Minimization method to use in scipy.optimize.minimize.
        n_times : int
            Number of sample times.
        jac_on : bool
            If jac_on=True, use the Jacobian of the energy cost function in
            scipy.optimize.minimize.

        Returns
        -------
        dist : float
            Length of the polynomial approximation of the geodesic.
        curve : array-like, shape=(n_times, dim)
            Polynomial approximation of the geodesic.
        velocity : array-like, shape=(n_times, dim)
            Velocity of the polynomial approximation of the geodesic.
        """

        def cost_fun(param):
            """Compute the energy of the polynomial curve defined by param.

            Parameters
            ----------
            param : array-like, shape=(degree - 1, dim)
                Parameters of the curve coordinates' polynomial functions of time.

            Returns
            -------
            energy : float
                Energy of the polynomial approximation of the geodesic.
            length : float
                Length of the polynomial approximation of the geodesic.
            curve : array-like, shape=(n_times, dim)
                Polynomial approximation of the geodesic.
            velocity : array-like, shape=(n_times, dim)
                Velocity of the polynomial approximation of the geodesic.
            """
            last_coef = end_point - initial_point - gs.sum(param, axis=0)
            coef = gs.vstack((initial_point, param, last_coef))

            t = gs.linspace(0.0, 1.0, n_times)
            t_curve = [t**i for i in range(degree + 1)]
            t_curve = gs.stack(t_curve)
            curve = gs.einsum("ij,ik->kj", coef, t_curve)

            t_velocity = [i * t ** (i - 1) for i in range(1, degree + 1)]
            t_velocity = gs.stack(t_velocity)
            velocity = gs.einsum("ij,ik->kj", coef[1:], t_velocity)

            if curve.min() < 0:
                return np.inf, np.inf, curve, np.nan

            velocity_sqnorm = self.squared_norm(vector=velocity, base_point=curve)
            length = gs.sum(velocity_sqnorm ** (1 / 2)) / n_times
            energy = gs.sum(velocity_sqnorm) / n_times
            return energy, length, curve, velocity

        def cost_jacobian(param):
            """Compute the jacobian of the cost function at polynomial curve.

            Parameters
            ----------
            param : array-like, shape=(degree - 1, dim)
                Parameters of the curve coordinates' polynomial functions of time.

            Returns
            -------
            jac : array-like, shape=(dim * (degree - 1),)
                Jacobian of the cost function at polynomial curve.
            """
            last_coef = end_point - initial_point - gs.sum(param, 0)
            coef = gs.vstack((initial_point, param, last_coef))

            t = gs.linspace(0.0, 1.0, n_times)
            t_position = [t**i for i in range(degree + 1)]
            t_position = gs.stack(t_position)
            position = gs.einsum("ij,ik->kj", coef, t_position)

            t_velocity = [i * t ** (i - 1) for i in range(1, degree + 1)]
            t_velocity = gs.stack(t_velocity)
            velocity = gs.einsum("ij,ik->kj", coef[1:], t_velocity)

            fac1 = gs.stack(
                [
                    k * t ** (k - 1) - degree * t ** (degree - 1)
                    for k in range(1, degree)
                ]
            )
            fac2 = gs.stack([t**k - t**degree for k in range(1, degree)])
            fac3 = (velocity * gs.polygamma(1, position)).T - gs.sum(
                velocity, 1
            ) * gs.polygamma(1, gs.sum(position, 1))
            fac4 = (velocity**2 * gs.polygamma(2, position)).T - gs.sum(
                velocity, 1
            ) ** 2 * gs.polygamma(2, gs.sum(position, 1))

            cost_jac = (
                2 * gs.einsum("ij,kj->ik", fac1, fac3)
                + gs.einsum("ij,kj->ik", fac2, fac4)
            ) / n_times
            return cost_jac.T.reshape(dim * (degree - 1))

        def f2minimize(x):
            """Compute function to minimize."""
            param = gs.transpose(x.reshape((dim, degree - 1)))
            res = cost_fun(param)
            return res[0]

        def jacobian(x):
            """Compute jacobian of the function to minimize."""
            param = gs.transpose(x.reshape((dim, degree - 1)))
            return cost_jacobian(param)

        dim = initial_point.shape[0]
        x0 = gs.ones(dim * (degree - 1))
        jac = jacobian if jac_on else None
        sol = minimize(f2minimize, x0, method=method, jac=jac)
        opt_param = sol.x.reshape((dim, degree - 1)).T
        _, dist, curve, velocity = cost_fun(opt_param)

        return dist, curve, velocity


class DirichletRandomVariable(ScipyMultivariateRandomVariable):
    """A Dirichlet random variable."""

    def __init__(self, space):
        pdf = lambda x, point: dirichlet.pdf(gs.transpose(x), point)
        super().__init__(space, dirichlet.rvs, pdf)
