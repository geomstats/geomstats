"""Affine connections."""

from scipy.optimize import minimize

import geomstats.backend as gs
import geomstats.errors
import geomstats.vectorization
from geomstats.integrator import integrate


N_STEPS = 10
EPSILON = 1e-3


class Connection:
    r"""Class for affine connections.

    Parameters
    ----------
    dim : int
        Dimension of the underlying manifold.
    default_point_type : str, {\'vector\', \'matrix\'}
        Point type.
        Optional, default: \'vector\'.
    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: \'intrinsic\'.
    """

    def __init__(
            self, dim, default_point_type='vector',
            default_coords_type='intrinsic'):
        geomstats.errors.check_integer(dim, 'dim')
        geomstats.errors.check_parameter_accepted_values(
            default_point_type, 'default_point_type', ['vector', 'matrix'])

        self.dim = dim
        self.default_point_type = default_point_type
        self.default_coords_type = default_coords_type

    def christoffels(self, base_point):
        """Christoffel symbols associated with the connection.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        gamma : array-like, shape=[..., dim, dim, dim]
            Christoffel symbols, with the covariant index on
            the first dimension.
        """
        raise NotImplementedError(
            'The Christoffel symbols are not implemented.')

    def connection(self, tangent_vec_a, tangent_vec_b, base_point):
        """Covariant derivative.

        Connection applied to `tangent_vector_b` in the direction of
        `tangent_vector_a`, both tangent at `base_point`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        """
        raise NotImplementedError(
            'connection is not implemented.')

    def geodesic_equation(self, position, velocity):
        """Compute the geodesic ODE associated with the connection.

        Parameters
        ----------
        velocity : array-like, shape=[..., dim]
            Tangent vector at the position.
        position : array-like, shape=[..., dim]
            Point on the manifold, the position at which to compute the
            geodesic ODE.

        Returns
        -------
        geodesic_ode : array-like, shape=[..., dim]
            Value of the vector field to be integrated at position.
        """
        gamma = self.christoffels(position)
        equation = gs.einsum(
            '...kij,...i->...kj', gamma, velocity)
        equation = - gs.einsum(
            '...kj,...j->...k', equation, velocity)
        return equation

    def exp(self, tangent_vec, base_point, n_steps=N_STEPS, step='euler',
            point_type=None):
        """Exponential map associated to the affine connection.

        Exponential map at base_point of tangent_vec computed by integration
        of the geodesic equation (initial value problem), using the
        christoffel symbols

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
        initial_state = (base_point, tangent_vec)
        flow, _ = integrate(
            self.geodesic_equation, initial_state, n_steps=n_steps, step=step)

        exp = flow[-1]
        return exp

    def log(self, point, base_point, n_steps=N_STEPS, step='euler'):
        """Compute logarithm map associated to the affine connection.

        Solve the boundary value problem associated to the geodesic equation
        using the Christoffel symbols and conjugate gradient descent.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point on the manifold.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_steps : int
            Number of discrete time steps to take in the integration.
            Optional, default: N_STEPS.
        step : str, {'euler', 'rk4'}
            Numerical scheme to use for integration.
            Optional, default: 'euler'.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        """
        def objective(velocity):
            """Define the objective function."""
            velocity = gs.array(velocity)
            velocity = gs.cast(velocity, dtype=base_point.dtype)
            velocity = gs.reshape(velocity, base_point.shape)
            delta = self.exp(velocity, base_point, n_steps, step) - point
            loss = 1. / self.dim * gs.sum(delta ** 2, axis=-1)
            return gs.sum(loss)

        objective_with_grad = gs.autograd.value_and_grad(objective)
        tangent_vec = gs.random.rand(*gs.flatten(base_point).shape)
        res = minimize(
            objective_with_grad, tangent_vec, method='L-BFGS-B', jac=True,
            options={'disp': False, 'maxiter': 25})

        tangent_vec = gs.array(res.x)
        tangent_vec = gs.reshape(tangent_vec, base_point.shape)
        return tangent_vec

    def _pole_ladder_step(self, base_point, next_point, base_shoot,
                          return_geodesics=False):
        """Compute one Pole Ladder step.

        One step of pole ladder scheme [LP2013a]_ using the geodesic to
        transport along as main_geodesic of the parallelogram.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold, from which to transport.
        next_point : array-like, shape=[..., dim]
            Point on the manifold, to transport to.
        base_shoot : array-like, shape=[..., dim]
            Point on the manifold, end point of the geodesics starting
            from the base point with initial speed to be transported.
        return_geodesics : bool, optional (defaults to False)
            Whether to return the geodesics of the
            construction.

        Returns
        -------
        next_step : dict of array-like and callable with following keys:
            next_tangent_vec : array-like, shape=[..., dim]
                Tangent vector at end point.
            end_point : array-like, shape=[..., dim]
                Point on the manifold, closes the geodesic parallelogram of the
                construction.
            geodesics : list of callable, len=3 (only if
            `return_geodesics=True`)
                Three geodesics of the construction.

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

        next_tangent_vec = - self.log(
            base_point=next_point, point=end_shoot)

        end_point = self.exp(
            base_point=next_point,
            tangent_vec=next_tangent_vec)

        geodesics = []
        if return_geodesics:
            main_geodesic = self.geodesic(
                initial_point=base_point,
                end_point=next_point)
            diagonal = self.geodesic(
                initial_point=mid_point,
                end_point=base_shoot)
            final_geodesic = self.geodesic(
                initial_point=next_point,
                end_point=end_shoot)
            geodesics = [main_geodesic, diagonal, final_geodesic]
        return {'next_tangent_vec': next_tangent_vec,
                'geodesics': geodesics,
                'end_point': end_point}

    def _schild_ladder_step(self, base_point, next_point, base_shoot,
                            return_geodesics=False):
        """Compute one Schild's Ladder step.

        One step of the Schild's ladder scheme [LP2013a]_ using the geodesic to
        transport along as one side of the parallelogram.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold, from which to transport.
        next_point : array-like, shape=[..., dim]
            Point on the manifold, to transport to.
        base_shoot : array-like, shape=[..., dim]
            Point on the manifold, end point of the geodesics starting
            from the base point with initial speed to be transported.
        return_geodesics : bool
            Whether to return points computed along each geodesic of the
            construction.
            Optional, default: False.

        Returns
        -------
        transported_tangent_vector : array-like, shape=[..., dim]
            Tangent vector at end point.
        end_point : array-like, shape=[..., dim]
            Point on the manifold, closes the geodesic parallelogram of the
            construction.

        References
        ----------
        .. [LP2013a] Marco Lorenzi, Xavier Pennec. Efficient Parallel Transport
         of Deformations in Time Series of Images: from Schild's to
         Pole Ladder. Journal of Mathematical Imaging and Vision, Springer
         Verlag, 2013,50 (1-2), pp.5-17. ⟨10.1007/s10851-013-0470-3⟩
        """
        mid_tangent_vector_to_shoot = 1. / 2. * self.log(
            base_point=base_shoot,
            point=next_point)

        mid_point = self.exp(
            base_point=base_shoot,
            tangent_vec=mid_tangent_vector_to_shoot)

        tangent_vector_to_shoot = - self.log(
            base_point=mid_point,
            point=base_point)

        end_shoot = self.exp(
            base_point=mid_point,
            tangent_vec=tangent_vector_to_shoot)

        next_tangent_vec = self.log(
            base_point=next_point, point=end_shoot)

        geodesics = []
        if return_geodesics:
            main_geodesic = self.geodesic(
                initial_point=base_point,
                end_point=next_point)
            diagonal = self.geodesic(
                initial_point=base_point,
                end_point=end_shoot)
            second_diagonal = self.geodesic(
                initial_point=base_shoot,
                end_point=next_point)
            final_geodesic = self.geodesic(
                initial_point=next_point,
                end_point=end_shoot)
            geodesics = [
                main_geodesic,
                diagonal,
                second_diagonal,
                final_geodesic]
        return {'next_tangent_vec': next_tangent_vec,
                'geodesics': geodesics,
                'end_point': end_shoot}

    def ladder_parallel_transport(
            self, tangent_vec_a, tangent_vec_b, base_point, n_steps=1,
            step='pole', **single_step_kwargs):
        """Approximate parallel transport using the pole ladder scheme.

        Approximate Parallel transport using either the pole ladder or the
        Schild's ladder scheme [LP2013b]_. Pole ladder is exact in symmetric
        spaces [GJSP2019]_ while Schild's ladder is a first order
        approximation. Both schemes are available any affine connection
        manifolds whose exponential and logarithm maps are implemented.
        `tangent_vec_a` is transported along the geodesic starting
        at the base_point with initial tangent vector `tangent_vec_b`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            Tangent vector at base point to transport.
        tangent_vec_b : array-like, shape=[..., dim]
            Tangent vector at base point, initial speed of the geodesic along
            which to transport.
        base_point : array-like, shape=[..., dim]
            Point on the manifold, initial position of the geodesic along
            which to transport.
        n_steps : int
            The number of pole ladder steps.
            Optional, default: 1.
        step : str, {'pole', 'schild'}
            The scheme to use for the construction of the ladder at each step.
            Optoinal, default: 'pole'.
        **single_step_kwargs : keyword arguments for the step functions

        Returns
        -------
        ladder : dict of array-like and callable with following keys
            transported_tangent_vector : array-like, shape=[..., dim]
                Approximation of the parallel transport of tangent vector a.
            trajectory : list of list of callable, len=n_steps
                List of lists containing the geodesics of the
                construction, only if `return_geodesics=True` in the step
                function. The geodesics are methods of the class connection.

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
        next_tangent_vec = gs.copy(tangent_vec_a) / n_steps
        methods = {'pole': self._pole_ladder_step,
                   'schild': self._schild_ladder_step}
        single_step = methods[step]
        base_shoot = self.exp(
            base_point=current_point, tangent_vec=next_tangent_vec)
        trajectory = []
        for i_point in range(n_steps):
            frac_tangent_vector_b = (i_point + 1) / n_steps * tangent_vec_b
            next_point = self.exp(
                base_point=base_point,
                tangent_vec=frac_tangent_vector_b)
            next_step = single_step(
                base_point=current_point,
                next_point=next_point,
                base_shoot=base_shoot,
                **single_step_kwargs)
            current_point = next_point
            base_shoot = next_step['end_point']
            trajectory.append(next_step['geodesics'])
        transported_tangent_vec = n_steps * next_step['next_tangent_vec']

        return {'transported_tangent_vec': transported_tangent_vec,
                'trajectory': trajectory}

    def riemannian_curvature(self, base_point):
        """Compute Riemannian curvature tensor associated with the connection.

        Parameters
        ----------
        base_point: array-like, shape=[..., dim]
            Point on the manifold.
        """
        raise NotImplementedError(
            'The Riemannian curvature tensor is not implemented.')

    def geodesic(self, initial_point,
                 end_point=None, initial_tangent_vec=None,
                 point_type=None):
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
        point_type : str, {'vector', 'matrix'}
            Point type.
            Optional, default: 'vector'.

        Returns
        -------
        path : callable
            Time parameterized geodesic curve. If a batch of initial
            conditions is passed, the output array's first dimension
            represents time, and the second corresponds to the different
            initial conditions.
        """
        if point_type is None:
            point_type = self.default_point_type
        geomstats.errors.check_parameter_accepted_values(
            point_type, 'point_type', ['vector', 'matrix'])

        if end_point is None and initial_tangent_vec is None:
            raise ValueError('Specify an end point or an initial tangent '
                             'vector to define the geodesic.')
        if end_point is not None:
            shooting_tangent_vec = self.log(
                point=end_point, base_point=initial_point)
            if initial_tangent_vec is not None:
                if not gs.allclose(shooting_tangent_vec, initial_tangent_vec):
                    raise RuntimeError(
                        'The shooting tangent vector is too'
                        ' far from the input initial tangent vector.')
            initial_tangent_vec = shooting_tangent_vec

        if point_type == 'vector':
            initial_point = gs.to_ndarray(initial_point, to_ndim=2)
            initial_tangent_vec = gs.to_ndarray(
                initial_tangent_vec, to_ndim=2)

        else:
            initial_point = gs.to_ndarray(initial_point, to_ndim=3)
            initial_tangent_vec = gs.to_ndarray(
                initial_tangent_vec, to_ndim=3)
        n_initial_conditions = initial_tangent_vec.shape[0]

        def path(t):
            """Generate parameterized function for geodesic curve.

            Parameters
            ----------
            t : array-like, shape=[n_points,]
                Times at which to compute points of the geodesics.
            """
            t = gs.array(t, gs.float32)
            t = gs.to_ndarray(t, to_ndim=1)
            if point_type == 'vector':
                tangent_vecs = gs.einsum(
                    'i,...k->...ik', t, initial_tangent_vec)
            else:
                tangent_vecs = gs.einsum(
                    'i,...kl->...ikl', t, initial_tangent_vec)

            points_at_time_t = [
                self.exp(tv, pt) for tv,
                pt in zip(tangent_vecs, initial_point)]
            points_at_time_t = gs.stack(points_at_time_t, axis=1)

            return points_at_time_t[:, 0] if n_initial_conditions == 1 else \
                points_at_time_t
        return path

    def torsion(self, base_point):
        """Compute torsion tensor associated with the connection.

        Parameters
        ----------
        base_point: array-like, shape=[..., dim]
            Point on the manifold.
        """
        raise NotImplementedError(
            'The torsion tensor is not implemented.')
