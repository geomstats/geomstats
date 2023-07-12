"""Implementation of sub-Riemannian geometry.

Lead author: Morten Pedersen.
"""

import geomstats.backend as gs


class SubRiemannianMetric:
    """Class for Sub-Riemannian metrics.

    This implementation assumes a bracket-generating distribution of constant dimension.

    Only one of the arguments 'cometric_matrix' and 'frame' can be different from
    None. If the frame is supplied, it is assumed orthonormal.

    Parameters
    ----------
    space : Manifold object
    cometric_matrix : callable
        Optional, default: 'None'

        The cometric matrix as a function of a point.

        You should pass :
            base_point : array-like, shape=[..., dim]
        It returns:
            _ : array-like, shape=[..., dim, dim]

    frame : callable
        Optional, default: 'None'

        Matrix representing the frame spanning the distribution,
        as a function of a point.

        You should pass:
            point : array-like, shape=[..., dim]
        It returns
            _ : array-like, shape=[..., dim, dist_dim]
                Frame field matrix. Each column is a vector field of the frame
                spanning the distribution.
    """

    def __init__(
        self,
        space,
        cometric_matrix=None,
        frame=None,
    ):
        if not bool(cometric_matrix is not None) ^ bool(frame is not None):
            raise ValueError(
                "Either 'cometric_matrix' or 'frame' must be passed, and not both."
            )

        if space.default_point_type != "vector":
            raise ValueError("Can only handle 'vector' point types.")

        self.cometric_matrix = cometric_matrix
        self.frame = frame

        self._def_type = "cometric" if self.frame is None else "frame"

    def sr_sharp(self, base_point, cotangent_vec):
        r"""Compute sub-Riemannian sharp map.

        This is the sub-Riemannian sharp map, mapping a covector at base_point to a
        tangent vector in the distribution subspace at base_point. For an orthonormal
        frame (F_i)_{i=1..dist_dim}, the sharp map is given by

        .. math::
                \sharp(q, p) = \sum_i^{dist_dim} p(F_i(q)) * F_i(q)

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        cotangent_vec : array-like, shape=[..., dim]
            Cotangent vector at `base_point`.

        Returns
        -------
        sr_sharp : array-like, shape=[..., dim]
            sub-Riemannian sharp of 'cotangent_vec' at 'base_point'
        """
        if self.frame is None:
            raise NotImplementedError(
                "The sub-Riemannian sharp map is only"
                " implemented when a frame is passed."
            )

        frame = self.frame(base_point)
        coefs = gs.einsum("...i,...ij->...j", cotangent_vec, frame)
        coefs_on_frame = gs.einsum("...j,...ij->...ij", coefs, frame)

        return gs.einsum("...ij->...i", coefs_on_frame)

    def _get_method(self, method_name):
        return getattr(self, f"_{method_name}_{self._def_type}")

    def _inner_coproduct_frame(self, cotangent_vec_a, cotangent_vec_b, base_point):
        sharp = self.sr_sharp(base_point=base_point, cotangent_vec=cotangent_vec_b)
        return gs.einsum("...i,...i -> ...", cotangent_vec_a, sharp)

    def _inner_coproduct_cometric(self, cotangent_vec_a, cotangent_vec_b, base_point):
        C_b = gs.einsum(
            "...ij,...j->...i", self.cometric_matrix(base_point), cotangent_vec_b
        )
        return gs.einsum("...i,...i->...", cotangent_vec_a, C_b)

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
        return self._get_method("inner_coproduct")(
            cotangent_vec_a, cotangent_vec_b, base_point
        )

    def _hamiltonian_frame(self, state):
        position, momentum = state

        inner_products = gs.einsum("...i,...ij->...j", momentum, self.frame(position))

        out = 0.5 * gs.einsum("...i,...i->...", inner_products, inner_products)
        if not gs.is_array(out):
            return gs.array(out)
        return out

    def _hamiltonian_cometric(self, state):
        position, momentum = state
        return 0.5 * self.inner_coproduct(momentum, momentum, position)

    def hamiltonian(self, state):
        r"""Compute the hamiltonian energy associated to the cometric.

        The Hamiltonian at state :math:`(q, p)` is defined by

        .. math::
                H(q, p) = \frac{1}{2} <p, p>_q

        where :math:`<\cdot, \cdot>_q` is the cometric at :math:`q`.

        Parameters
        ----------
        state : array-like, shape=[[..., dim], [..., dim]]
            The first array in 'state' contains positions, the second contains
            covectors (momentums).

        Returns
        -------
        energy :  float
            Hamiltonian energy at `state`.
        """
        return self._get_method("hamiltonian")(state)

    @staticmethod
    def symp_grad(hamiltonian):
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

        def H_sum(state):
            """Sum each value of the Hamiltonian (relevant for vectorized input)."""
            # TODO: nice trick to vectorize the gradient
            return gs.sum(hamiltonian(state))

        # TODO: vectorized grad
        value_and_grad = gs.autodiff.value_and_grad(H_sum)

        def vector(x):
            """Compute symplectic gradient at x."""
            # TODO: update after add vectorized grad to backend
            _, grad = value_and_grad(x)
            h_q = grad[0]
            h_p = grad[1]
            return gs.array([h_p, -h_q])

        return vector

    def _symp_euler_frame(self, hamiltonian, step_size):
        def step(state):
            position, momentum = state
            dq = self.sr_sharp(base_point=position, cotangent_vec=momentum)
            y = gs.array([position + step_size * dq, momentum])
            _, dp = self.symp_grad(hamiltonian)(y)
            return gs.array([position + step_size * dq, momentum + step_size * dp])

        return step

    def _symp_euler_cometric(self, hamiltonian, step_size):
        def step(state):
            r"""Compute an integration step from state."""
            position, momentum = state
            dq, _ = self.symp_grad(hamiltonian)(state)
            y = gs.array([position + step_size * dq, momentum])
            _, dp = self.symp_grad(hamiltonian)(y)
            return gs.array([position + step_size * dq, momentum + step_size * dp])

        return step

    def symp_euler(self, hamiltonian, step_size):
        r"""Compute a function which calculates a step of symplectic euler integration.

        Parameters
        ----------
        hamiltonian : callable
            The hamiltonian function from the tangent bundle to the reals.
        step_size : float
            Step size of the symplectic euler step

        Returns
        -------
        step : callable
            Given a state, 'step' returns the next symplectic euler step
        """
        return self._get_method("symp_euler")(hamiltonian, step_size)

    @staticmethod
    def iterate(func, n_steps):
        r"""Construct a function which iterates another function n_steps times.

        Parameters
        ----------
        func : callable
            A function which calculates the next step of a
            sequence to be calculated.
        n_steps : int
            The number of times to iterate func.

        Returns
        -------
        flow : callable
            Given a state, 'flow' returns a sequence with n_steps
            iterations of func.
        """

        def flow(x):
            r"""Compute flow starting at x."""
            xs = [x]
            for i in range(n_steps):
                xs.append(func(xs[i]))
            return gs.array(xs)

        return flow

    def symp_flow(self, hamiltonian, end_time=1.0, n_steps=20):
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
            Given a state, 'symp_flow' returns a
            sequence with n_steps iterations of func.
        """
        step = self.symp_euler
        step_size = end_time / n_steps
        return self.iterate(step(hamiltonian, step_size), n_steps)

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
        point_type : str, {'vector', 'matrix'}
            Type of representation used for points.
            Optional, default: None.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point on the manifold.
        """
        # TODO: add log (caveat in docstrings)
        # TODO: add distance (based on log - also caveat)
        base_point = gs.broadcast_to(base_point, cotangent_vec.shape)

        # TODO: integrate; allow euler
        initial_state = gs.stack([base_point, cotangent_vec])

        flow = self.symp_flow(self.hamiltonian, n_steps=n_steps)

        return flow(initial_state)[-1][0]

    def geodesic(self, initial_point, initial_cotangent_vec, n_steps=20):
        """Generate parameterized function for the normal geodesic curve.

        Normal geodesic curve defined by an initial point and an initial
        cotangent vector.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Point on the manifold, initial point of the geodesic.
        initial_cotangent_vec : array-like, shape=[..., dim],
            Cotangent vector at base point, the initial speed of the geodesics.

        Returns
        -------
        path : callable
            Time parameterized normal geodesic curve. If a batch of initial
            conditions is passed, the output array's first dimension
            represents the different initial conditions, and the second
            corresponds to time.
        """
        if initial_cotangent_vec is None:
            raise ValueError(
                "Specify an initial cotangent " "vector to define the geodesic."
            )

        initial_point = gs.to_ndarray(initial_point, to_ndim=2)
        initial_cotangent_vec = gs.to_ndarray(initial_cotangent_vec, to_ndim=2)

        n_initial_conditions = initial_cotangent_vec.shape[0]

        if n_initial_conditions > 1 and len(initial_point) == 1:
            initial_point = gs.stack([initial_point[0]] * n_initial_conditions)

        def path(t):
            """Generate parameterized function for geodesic curve.

            Parameters
            ----------
            t : array-like, shape=[n_points,]
                Times at which to compute points of the geodesics.
            """
            t = gs.array(t)
            t = gs.cast(t, initial_cotangent_vec.dtype)
            t = gs.to_ndarray(t, to_ndim=1)

            cotangent_vecs = gs.einsum("i,...k->...ik", t, initial_cotangent_vec)

            points_at_time_t = [
                self.exp(tv, pt, n_steps=n_steps)
                for tv, pt in zip(cotangent_vecs, initial_point)
            ]
            points_at_time_t = gs.stack(points_at_time_t, axis=0)

            return (
                points_at_time_t[0] if n_initial_conditions == 1 else points_at_time_t
            )

        return path
