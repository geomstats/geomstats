"""Implementation of sub-Riemannian geometry."""

import abc

import geomstats.backend as gs
import geomstats.geometry as geometry
from autograd.scipy.integrate import odeint


class SubRiemannianMetric(abc.ABC):
    """Class for Sub-Riemannian metrics.

    This implementation assumes a distribution of constant dimension.

    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    dist_dim : int
        Dimension of the distribution
    default_point_type : str, {'vector', 'matrix'}
        Point type.
        Optional, default: 'vector'.
    """

    def __init__(self, dim, dist_dim, default_point_type="vector"):
        self.dim = dim
        self.dist_dim = dist_dim
        self.default_point_type = default_point_type

    # def metric_matrix(self, base_point):
    #     """Metric matrix at the tangent space at a base point.

    #     This is a sub-Riemannian metric, so it is assumed to satisfy the conditions
    #     of an inner product only on each distribution subspace.

    #     Parameters
    #     ----------
    #     base_point : array-like, shape=[..., dim]
    #         Base point.
    #         Optional, default: None.

    #     Returns
    #     -------
    #     _ : array-like, shape=[..., dim, dim]
    #         Inner-product matrix.
    #     """
    #     raise NotImplementedError(
    #         "The computation of the metric matrix" " is not implemented."
    #     )

    # @abc.abstractmethod
    # def frame(self, point):
    #     """Frame field for the distribution.

    #     The frame field spans the distribution at 'point'.The frame field is
    #     represented as a matrix, whose columns are the frame field vectors.

    #     Parameters
    #     ----------
    #     point : array-like, shape=[..., dim]
    #         point.
    #         Optional, default: None.

    #     Returns
    #     -------
    #     _ : array-like, shape=[..., dim, dist_dim]
    #         Frame field matrix.
    #     """

    # def cometric_sub_matrix(self, basepoint):
    #     """Cometric  sub matrix of dimension dist_dim x dist_dim.

    #     Let {X_i}, i = 1, .., dist_dim, be an arbitrary frame for the distribution
    #     and let g be the sub-Riemannian metric. Then cometric_sub_matrix is the
    #     matrix given by the inverse of the matrix g_ij = g(X_i, X_j),
    #     where i,j = 1, .., dist_dim.

    #     Parameters
    #     ----------
    #     base_point : array-like, shape=[..., dim]
    #         Base point.
    #         Optional, default: None.

    #     Returns
    #     -------
    #     _ : array-like, shape=[..., dist_dim, dist_dim]
    #         Cometric submatrix.
    #     """
    #     raise NotImplementedError(
    #         "The computation of the cometric submatrix" " is not implemented."
    #     )

    @abc.abstractmethod
    def cometric_matrix(self, base_point=None):
        """Inner co-product matrix at the cotangent space at a base point.

        This represents the cometric matrix, i.e. the inverse of the
        metric matrix.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inverse of inner-product matrix.
        """

    def inner_coproduct(self, cotangent_vec_a, cotangent_vec_b, base_point):
        """Compute inner coproduct between two cotangent vectors at base point.

        This is the inner product associated to the cometric matrix.

        Parameters
        ----------
        cotangent_vec_a : array-like, shape=[..., dim]
            Cotangent vector at `base_point`.
        cotangent_vet_b : array-like, shape=[..., dim]
            Cotangent vector at `base_point`.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        inner_coproduct : float
            Inner coproduct between the two cotangent vectors.
        """
        vector_2 = gs.einsum(
            "...ij,...j->...i", self.cometric_matrix(base_point), cotangent_vec_b
        )
        inner_coproduct = gs.einsum("...i,...i->...", cotangent_vec_a, vector_2)
        return inner_coproduct

    def hamiltonian(self, state):
        r"""Compute the hamiltonian energy associated to the cometric.

        The Hamiltonian at state :math: `(q, p)` is defined by
        .. math:
                H(q, p) = \frac{1}{2} <p, p>_q
        where :math: `<\cdot, \cdot>_q` is the cometric at :math: `q`.

        Parameters
        ----------
        state : tuple of arrays
            Position and momentum variables. The position is a point on the
            manifold, while the momentum is cotangent vector.

        Returns
        -------
        energy : float
            Hamiltonian energy at `state`.
        """
        position, momentum = state
        return 1.0 / 2 * self.inner_coproduct(momentum, momentum, position)

    def symp_grad(self, hamiltonian):
        def vector(x):
            _, grad = gs.autodiff.value_and_grad(self.hamiltonian)(x)
            h_q = grad[0]
            h_p = grad[1]
            return gs.array([h_p, -h_q])

        return vector

    def symp_euler(self, hamiltonian, step_size):
        def step(state):
            position, momentum = state
            dq, _ = self.symp_grad(self.hamiltonian)(state)
            y = gs.array([position + dq, momentum])
            _, dp = self.symp_grad(self.hamiltonian)(y)
            return gs.array([position + step_size * dq, momentum + step_size * dp])

        return step


    @staticmethod
    def iterate(func, n_steps):
        def flow(x):
            xs = [x]
            for i in range(n_steps):
                xs.append(func(xs[i]))
            return gs.array(xs)

        return flow

    def symp_flow(self, hamiltonian, end_time=1.0, n_steps=20):
            step = self.symp_euler
            step_size = end_time / n_steps
            return self.iterate(step(self.hamiltonian, step_size), n_steps)
    
    def exp(
        self,
        cotangent_vec,
        base_point,
        n_steps=20,
        point_type=None,
        **kwargs
    ):
        """Exponential map associated to the affine connection.

        Exponential map at base_point of tangent_vec computed by integration
        of the geodesic equation (initial value problem), using the
        christoffel symbols.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_steps : int
            Number of discrete time steps to take in the integration.
            Optional, default: N_STEPS.
        step : str, {'euler', 'rk4'}
            The numerical scheme to use for integration.
            Optional, default: 'euler'.
        point_type : str, {'vector', 'matrix'}
            Type of representation used for points.
            Optional, default: None.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point on the manifold.
        """
        initial_state = gs.stack([base_point, cotangent_vec])

        flow = self.symp_flow(self.hamiltonian, n_steps=n_steps)

        exp = flow(initial_state)[-1][0]
        return exp

    

    




    
