"""Affine connections."""

import autograd
from scipy.optimize import minimize

import geomstats.backend as gs
from geomstats.integrator import integrate


N_STEPS = 10
EPSILON = 1e-3


class Connection(object):
    """Classe for affine connections."""

    def __init__(self, dimension):
        self.dimension = dimension

    def christoffels(self, base_point):
        """Christoffel symbols associated with the connection.

        Parameters
        ----------
        base_point : array-like, shape=[n_samples, dimension]

        Returns
        -------
        gamma : array-like, shape=[n_samples, dimension, dimension, dimension]
        """
        raise NotImplementedError(
            'The Christoffel symbols are not implemented.')

    def connection(self, tangent_vector_a, tangent_vector_b, base_point):
        """Covariant derivative.

        Connection applied to `tangent_vector_b` in the direction of
        `tangent_vector_a`, both tangent at `base_point`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, dimension]
        tangent_vec_b : array-like, shape=[n_samples, dimension]
        base_point : array-like, shape=[n_samples, dimension]
        """
        raise NotImplementedError(
            'connection is not implemented.')

    def geodesic_equation(self, position, velocity):
        """Compute the geodesic ODE associated with the connection.

        Parameters
        ----------
        velocity : array-like, shape=[n_samples, dimension]
        position : array-like, shape=[n_samples, dimension]
            the position at which to compute the geodesic ODE

        Returns
        -------
        geodesic_ode : array-like, shape=[n_samples, dimension]
            value of the vector field to be integrated at position
        """
        gamma = self.christoffels(position)
        return - gs.einsum('...kij,...i,...j->...k', gamma, velocity,
                           velocity)

    def exp(self, tangent_vec, base_point, n_steps=N_STEPS, step='euler'):
        """Exponential map associated to the affine connection.

        Exponential map at base_point of tangent_vec computed by integration
        of the geodesic equation (initial value problem), using the
        christoffel symbols

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, dimension]
        base_point : array-like, shape=[n_samples, dimension]
        n_steps : int
            the number of discrete time steps to take in the integration
        step : str, {'euler', 'rk4'}
            the numerical scheme to use for integration

        Returns
        -------
        exp : array-like, shape=[n_samples, dimension]
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        initial_state = (base_point, tangent_vec)
        flow, _ = integrate(
            self.geodesic_equation, initial_state, n_steps=n_steps, step=step)
        return flow[-1]

    def log(self, point, base_point, n_steps=N_STEPS, step='euler'):
        """Compute logarithm map associated to the affine connection.

        Solve the boundary value problem associated to the geodesic equation
        using the Christoffel symbols and conjugate gradient descent.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
        base_point : array-like, shape=[n_samples, dimension]
        n_steps : int
            the number of discrete time steps to take in the integration
        step : str, {'euler', 'rk4'}
            the numerical scheme to use for integration

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, dimension]
        """
        point = gs.to_ndarray(point, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        n_samples = len(base_point)

        def objective(velocity):
            """Define the objective function."""
            velocity = velocity.reshape(base_point.shape)
            delta = self.exp(velocity, base_point, n_steps, step) - point
            loss = 1. / self.dimension * gs.sum(delta ** 2, axis=1)
            return 1. / n_samples * gs.sum(loss)

        objective_grad = autograd.elementwise_grad(objective)

        def objective_with_grad(velocity):
            """Create helpful objective func wrapper for autograd comp."""
            return objective(velocity), objective_grad(velocity)

        tangent_vec = gs.random.rand(base_point.size)
        res = minimize(
            objective_with_grad, tangent_vec, method='L-BFGS-B', jac=True,
            options={'disp': False, 'maxiter': 25})

        tangent_vec = res.x
        tangent_vec = gs.reshape(tangent_vec, base_point.shape)
        return tangent_vec

    def pole_ladder_step(self, base_point, next_point, base_shoot):
        """Compute one Pole Ladder step.

        One step of pole ladder scheme [LP2013a]_ using the geodesic to
        transport along as diagonal of the parallelogram.

        Parameters
        ----------
        base_point : array-like
            shape=[n_samples, dimension] or shape=[1, dimension]
        next_point : array-like
            shape=[n_samples, dimension] or shape=[1, dimension]
        base_shoot : array-like
            shape=[n_samples, dimension] or shape=[1, dimension]

        Returns
        -------
        transported_tangent_vector : array-like
            shape=[n_samples, dimension] or shape=[1, dimension]
        end_point : array-like
            shape=[n_samples, dimension] or shape=[1, dimension]

        References
        ----------
        .. [LP2013a] Marco Lorenzi, Xavier Pennec. Efficient Parallel Transport
         of Deformations in Time Series of Images: from Schild's to
         Pole Ladder. Journal of Mathematical Imaging and Vision, Springer
         Verlag, 2013,50 (1-2), pp.5-17. ⟨10.1007/s10851-013-0470-3⟩
        """
        mid_tangent_vector_to_shoot = 1. / 2. * self.log(
            base_point=base_point,
            point=next_point)

        mid_point = self.exp(
            base_point=base_point,
            tangent_vec=mid_tangent_vector_to_shoot)

        tangent_vector_to_shoot = - self.log(
            base_point=mid_point,
            point=base_shoot)

        end_shoot = self.exp(
            base_point=mid_point,
            tangent_vec=tangent_vector_to_shoot)

        transported_tangent_vector = - self.log(
            base_point=next_point, point=end_shoot)

        end_point = self.exp(
            base_point=next_point,
            tangent_vec=transported_tangent_vector)

        return transported_tangent_vector, end_point

    def pole_ladder_parallel_transport(
            self, tangent_vec_a, tangent_vec_b, base_point, n_steps=1):
        """Approximate parallel transport using the pole ladder scheme.

        Approximate Parallel transport using the pole ladder scheme [LP2013b]_
        [GJSP2019]_. `tangent_vec_a` is transported along the geodesic starting
        at the base_point with initial tangent vector `tangent_vec_b`.

        Returns a tangent vector at the point
        exp_(`base_point`)(`tangent_vec_b`).

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]
        tangent_vec_b : array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]
        base_point : array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        n_steps: int
            the number of pole ladder steps

        Returns
        -------
        transported_tangent_vector : array-like, shape=[n_samples, dimension]
                                                or shape=[1, dimension]

        References
        ----------
        .. [LP2013b] Marco Lorenzi, Xavier Pennec. Efficient Parallel Transpor
          of Deformations in Time Series of Images: from Schild's to
          Pole Ladder.Journal of Mathematical Imaging and Vision, Springer
          Verlag, 2013, 50 (1-2), pp.5-17. ⟨10.1007/s10851-013-0470-3⟩

        .. [GJSP2019] N. Guigui, Shuman Jia, Maxime Sermesant, Xavier Pennec.
          Symmetric Algorithmic Components for Shape Analysis with
          Diffeomorphisms. GSI 2019, Aug 2019, Toulouse, France. pp.10.
          ⟨hal-02148832⟩
        """
        current_point = gs.copy(base_point)
        transported_tangent_vector = gs.copy(tangent_vec_a)
        base_shoot = self.exp(base_point=current_point,
                              tangent_vec=transported_tangent_vector)
        for i_point in range(0, n_steps):
            frac_tangent_vector_b = (i_point + 1) / n_steps * tangent_vec_b
            next_point = self.exp(
                base_point=base_point,
                tangent_vec=frac_tangent_vector_b)
            transported_tangent_vector, base_shoot = self.pole_ladder_step(
                base_point=current_point,
                next_point=next_point,
                base_shoot=base_shoot)
            current_point = next_point

        return transported_tangent_vector

    def riemannian_curvature(self, base_point):
        """Compute Riemannian curvature tensor associated with the connection.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        raise NotImplementedError(
            'The Riemannian curvature tensor is not implemented.')

    def geodesic(self, initial_point,
                 end_point=None, initial_tangent_vec=None,
                 point_type='vector'):
        """Generate parameterized function for the geodesic curve.

        Geodesic curve defined by either:
        - an initial point and an initial tangent vector,
        - an initial point and an end point.

        Parameters
        ----------
        initial_point
        end_point
        initial_tangent_vec
        point_type

        Returns
        -------
        path : callable
            the time parameterized geodesic curve.
        """
        point_ndim = 1
        if point_type == 'matrix':
            point_ndim = 2

        initial_point = gs.to_ndarray(
            initial_point, to_ndim=point_ndim + 1)

        if end_point is None and initial_tangent_vec is None:
            raise ValueError('Specify an end point or an initial tangent '
                             'vector to define the geodesic.')
        if end_point is not None:
            end_point = gs.to_ndarray(
                end_point, to_ndim=point_ndim + 1)
            shooting_tangent_vec = self.log(point=end_point,
                                            base_point=initial_point)
            if initial_tangent_vec is not None:
                assert gs.allclose(shooting_tangent_vec, initial_tangent_vec)
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = gs.array(initial_tangent_vec)
        initial_tangent_vec = gs.to_ndarray(
            initial_tangent_vec, to_ndim=point_ndim + 1)

        def path(t):
            """Generate parameterized function for geodesic curve."""
            t = gs.cast(t, gs.float32)
            t = gs.to_ndarray(t, to_ndim=1)
            t = gs.to_ndarray(t, to_ndim=2, axis=1)
            new_initial_point = gs.to_ndarray(
                initial_point,
                to_ndim=point_ndim + 1)
            new_initial_tangent_vec = gs.to_ndarray(
                initial_tangent_vec,
                to_ndim=point_ndim + 1)

            if point_type == 'vector':
                tangent_vecs = gs.einsum(
                    'il,nk->ik',
                    t,
                    new_initial_tangent_vec)
            elif point_type == 'matrix':
                tangent_vecs = gs.einsum(
                    'il,nkm->ikm',
                    t,
                    new_initial_tangent_vec)

            point_at_time_t = self.exp(
                tangent_vec=tangent_vecs,
                base_point=new_initial_point)
            return point_at_time_t

        return path

    def torsion(self, base_point):
        """Compute torsion tensor associated with the connection.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        raise NotImplementedError(
            'The torsion tensor is not implemented.')
