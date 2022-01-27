"""Sub-Riemannian metrics.

Lead author: Morten Pedersen.
"""

import abc

import geomstats.backend as gs


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

    def metric_matrix(self, base_point):
        """Metric matrix at the tangent space at a base point.

        This is a sub-Riemannian metric, so it is assumed to satisfy the conditions
        of an inner product only on each distribution subspace.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        _ : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        raise NotImplementedError(
            "The computation of the metric matrix is not implemented."
        )

    def frame(self, point):
        """Frame field for the distribution.

        The frame field spans the distribution at 'point'.The frame field is
        represented as a matrix, whose columns are the frame field vectors.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            point.
            Optional, default: None.

        Returns
        -------
        _ : array-like, shape=[..., dim, dist_dim]
            Frame field matrix.
        """
        raise NotImplementedError("The frame field is not implemented.")

    def cometric_sub_matrix(self, base_point):
        """Cometric  sub matrix of dimension dist_dim x dist_dim.

        Let {X_i}, i = 1, .., dist_dim, be an arbitrary frame for the distribution
        and let g be the sub-Riemannian metric. Then cometric_sub_matrix is the
        matrix given by the inverse of the matrix g_ij = g(X_i, X_j),
        where i,j = 1, .., dist_dim.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        _ : array-like, shape=[..., dist_dim, dist_dim]
            Cometric submatrix.
        """
        raise NotImplementedError(
            "The computation of the cometric submatrix is not implemented."
        )

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
        cotangent_vec_b : array-like, shape=[..., dim]
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
            Position and momentum variables. Position is a point on the
            manifold, while the momentum is cotangent vector.

        Returns
        -------
        energy : float
            Hamiltonian energy at `state`.
        """
        position, momentum = state
        return 1.0 / 2 * self.inner_coproduct(momentum, momentum, position)

    def symp_grad(self):
        r"""Compute the symplectic gradient of the Hamiltonian.

        Parameters
        ----------
        hamiltonian : callable
            The hamiltonian function from the tangent bundle to the reals.

        Returns
        -------
        vector : array-like, shape=[, 2*dim]
            The symplectic gradient of the Hamiltonian.
        """

        def vector(state):
            """Return symplectic gradient of Hamiltonian at state."""
            _, grad = gs.autodiff.value_and_grad(self.hamiltonian)(state)
            h_q = grad[0]
            h_p = grad[1]
            return gs.array([h_p, -h_q])

        return vector

    def symp_euler(self, step_size):
        """Compute a function which calculates a step of symplectic euler integration.

        The output function computes a symplectic euler step of the Hamiltonian system
        of equations associated with the cometric and obtained by the method
        :meth:`~sub_remannian_metric.SubRiemannianMetric.symp_grad`.

        Parameters
        ----------
        step_size : float
            Step size of the symplectic euler step.

        Returns
        -------
        step : callable
            Given a state, 'step' returns the next symplectic euler step.
        """

        def step(state):
            """Return the next symplectic euler step from state."""
            position, momentum = state
            dq, _ = self.symp_grad()(state)
            y = gs.array([position + dq, momentum])
            _, dp = self.symp_grad()(y)
            return gs.array([position + step_size * dq, momentum + step_size * dp])

        return step

    @staticmethod
    def iterate(func, n_steps):
        r"""Construct a function which iterates another function n_steps times.

        Parameters
        ----------
        func : callable
            A function which calculates the next step of a sequence to be calculated.
        n_steps : int
            The number of times to iterate func.

        Returns
        -------
        flow : callable
            Given a state, 'flow' returns a sequence with n_steps iterations of func.
        """

        def flow(x):
            """Return n_steps iterations of func from x."""
            xs = [x]
            for i in range(n_steps):
                xs.append(func(xs[i]))
            return gs.array(xs)

        return flow

    def symp_flow(self, end_time=1.0, n_steps=20):
        r"""Compute the symplectic flow of the hamiltonian.

        Parameters
        ----------
        hamiltonian : callable
            The hamiltonian function from the tangent bundle to the reals.
        end_time : float
            The last time point until which to calculate the flow.
        n_steps : int
            Number of integration steps.

        Returns
        -------
        _ : array-like, shape[,n_steps]
            Given a state, 'symp_flow' returns a sequence with
            n_steps iterations of func.
        """
        step = self.symp_euler
        step_size = end_time / n_steps
        return self.iterate(step(step_size), n_steps)

    def exp(self, cotangent_vec, base_point, n_steps=20, **kwargs):
        """Exponential map associated to the cometric.

        Exponential map at base_point of cotangent_vec computed by integration
        of the Hamiltonian equation (initial value problem), using the cometric.
        In the Riemannian case this yields the exponential associated to the
        Levi-Civita connection of the metric (the inverse of the cometric).

        Parameters
        ----------
        cotangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_steps : int
            Number of discrete time steps to take in the integration.
            Optional, default: N_STEPS.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point on the manifold.
        """
        initial_state = gs.stack([base_point, cotangent_vec])

        flow = self.symp_flow(n_steps=n_steps)

        return flow(initial_state)[-1][0]
