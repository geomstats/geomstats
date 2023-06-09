"""Affine connections.

Lead author: Nicolas Guigui.
"""

from abc import ABC

import geomstats.backend as gs
import geomstats.errors


def _check_log_solver(connection):
    if not hasattr(connection, "log_solver"):
        raise ValueError(
            "Requires `self.log_solver`. "
            "Check `geomstats.numerics.geodesic` for available solvers."
        )


def _check_exp_solver(connection):
    if not hasattr(connection, "exp_solver"):
        raise ValueError(
            "Requires `self.exp_solver`. "
            "Check `geomstats.numerics.geodesic` for available solvers."
        )


class Connection(ABC):
    r"""Class for affine connections.

    Parameters
    ----------
    space : Manifold object
        M in the tuple (M, g).
    """

    def __init__(self, space):
        self._space = space

    def christoffels(self, base_point):
        """Christoffel symbols associated with the connection.

        The contravariant index is on the first dimension.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        gamma : array-like, shape=[..., dim, dim, dim]
            Christoffel symbols, with the contravariant index on
            the first dimension.
        """
        raise NotImplementedError("The Christoffel symbols are not implemented.")

    def geodesic_equation(self, state, _time):
        """Compute the geodesic ODE associated with the connection.

        Parameters
        ----------
        state : array-like, shape=[..., dim]
            Tangent vector at the position.
        _time : array-like, shape=[..., dim]
            Point on the manifold, the position at which to compute the
            geodesic ODE.

        Returns
        -------
        geodesic_ode : array-like, shape=[..., dim]
            Value of the vector field to be integrated at position.
        """
        position, velocity = state
        gamma = self.christoffels(position)
        equation = gs.einsum("...kij,...i->...kj", gamma, velocity)
        equation = -gs.einsum("...kj,...j->...k", equation, velocity)
        return gs.stack([velocity, equation])

    def exp(self, tangent_vec, base_point):
        """Exponential map associated to the affine connection.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point on the manifold.
        """
        _check_exp_solver(self)
        return self.exp_solver.exp(self._space, tangent_vec, base_point)

    def log(self, point, base_point):
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
        max_iter
        verbose
        tol

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point.
        """
        _check_log_solver(self)
        return self.log_solver.log(self._space, point, base_point)

    def _pole_ladder_step(
        self, base_point, next_point, base_shoot, return_geodesics=False, **kwargs
    ):
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
        mid_tangent_vector_to_shoot = (
            1.0 / 2.0 * self.log(base_point=base_point, point=next_point, **kwargs)
        )

        mid_point = self.exp(
            base_point=base_point, tangent_vec=mid_tangent_vector_to_shoot, **kwargs
        )

        tangent_vector_to_shoot = -self.log(
            base_point=mid_point, point=base_shoot, **kwargs
        )

        end_shoot = self.exp(
            base_point=mid_point, tangent_vec=tangent_vector_to_shoot, **kwargs
        )

        geodesics = []
        if return_geodesics:
            main_geodesic = self.geodesic(
                initial_point=base_point, end_point=next_point
            )
            diagonal = self.geodesic(initial_point=mid_point, end_point=base_shoot)
            final_geodesic = self.geodesic(
                initial_point=next_point, end_point=end_shoot
            )
            geodesics = [main_geodesic, diagonal, final_geodesic]
        return {"geodesics": geodesics, "end_point": end_shoot}

    def _schild_ladder_step(
        self, base_point, next_point, base_shoot, return_geodesics=False, **kwargs
    ):
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
        mid_tangent_vector_to_shoot = (
            1.0 / 2.0 * self.log(base_point=base_shoot, point=next_point, **kwargs)
        )

        mid_point = self.exp(
            base_point=base_shoot, tangent_vec=mid_tangent_vector_to_shoot, **kwargs
        )

        tangent_vector_to_shoot = -self.log(
            base_point=mid_point, point=base_point, **kwargs
        )

        end_shoot = self.exp(
            base_point=mid_point, tangent_vec=tangent_vector_to_shoot, **kwargs
        )

        geodesics = []
        if return_geodesics:
            main_geodesic = self.geodesic(
                initial_point=base_point, end_point=next_point
            )
            diagonal = self.geodesic(initial_point=base_point, end_point=end_shoot)
            second_diagonal = self.geodesic(
                initial_point=base_shoot, end_point=next_point
            )
            final_geodesic = self.geodesic(
                initial_point=next_point, end_point=end_shoot
            )
            geodesics = [main_geodesic, diagonal, second_diagonal, final_geodesic]
        return {"geodesics": geodesics, "end_point": end_shoot}

    def ladder_parallel_transport(
        self,
        tangent_vec,
        base_point,
        direction,
        n_rungs=1,
        scheme="pole",
        alpha=1,
        **single_step_kwargs,
    ):
        """Approximate parallel transport using the pole ladder scheme.

        Approximate Parallel transport using either the pole ladder or the
        Schild's ladder scheme [LP2013b]_. Pole ladder is exact in symmetric
        spaces and of order two in general while Schild's ladder is a first
        order approximation [GP2020]_. Both schemes are available on any affine
        connection manifolds whose exponential and logarithm maps are
        implemented. `tangent_vec` is transported along the geodesic starting
        at the base_point with initial tangent vector `direction`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point to transport.
        direction : array-like, shape=[..., dim]
            Tangent vector at base point, initial speed of the geodesic along
            which to transport.
        base_point : array-like, shape=[..., dim]
            Point on the manifold, initial position of the geodesic along
            which to transport.
        n_rungs : int
            Number of steps of the ladder.
            Optional, default: 1.
        scheme : str, {'pole', 'schild'}
            The scheme to use for the construction of the ladder at each step.
            Optional, default: 'pole'.
        alpha : float
            Exponent for the scaling of the vector to transport. Must be
            greater or equal to 1, 2 is optimal. See [GP2020]_.
            Optional, default: 2
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
        .. [LP2013b] Lorenzi, Marco, and Xavier Pennec. “Efficient Parallel
            Transport of Deformations in Time Series of Images: From Schild to
            Pole Ladder.” Journal of Mathematical Imaging and Vision 50, no. 1
            (September 1, 2014): 5–17.
            https://doi.org/10.1007/s10851-013-0470-3.

        .. [GP2020] Guigui, Nicolas, and Xavier Pennec. “Numerical Accuracy
            of Ladder Schemes for Parallel Transport on Manifolds.”
            Foundations of Computational Mathematics, June 18, 2021.
            https://doi.org/10.1007/s10208-021-09515-x.
        """
        geomstats.errors.check_integer(n_rungs, "n_rungs")
        if alpha < 1:
            raise ValueError("alpha must be greater or equal to one")
        current_point = base_point
        next_tangent_vec = tangent_vec / (n_rungs**alpha)
        methods = {"pole": self._pole_ladder_step, "schild": self._schild_ladder_step}
        single_step = methods[scheme]
        base_shoot = self.exp(base_point=current_point, tangent_vec=next_tangent_vec)
        trajectory = []
        for i_point in range(n_rungs):
            frac_tan_vector_b = (i_point + 1) / n_rungs * direction
            next_point = self.exp(base_point=base_point, tangent_vec=frac_tan_vector_b)
            next_step = single_step(
                base_point=current_point,
                next_point=next_point,
                base_shoot=base_shoot,
                **single_step_kwargs,
            )
            current_point = next_point
            base_shoot = next_step["end_point"]
            trajectory.append(next_step["geodesics"])
        transported_tangent_vec = self.log(base_shoot, current_point)
        if n_rungs % 2 == 1 and scheme == "pole":
            transported_tangent_vec *= -1.0
        transported_tangent_vec *= n_rungs**alpha
        return {
            "transported_tangent_vec": transported_tangent_vec,
            "end_point": current_point,
            "trajectory": trajectory,
        }

    def riemann_tensor(self, base_point):
        r"""Compute Riemannian tensor at base_point.

        In the literature the riemannian curvature tensor is noted :math:`R_{ijk}^l`.

        Following tensor index convention (ref. Wikipedia), we have:
        :math:`R_{ijk}^l = dx^l(R(X_j, X_k)X_i)`

        which gives :math:`R_{ijk}^lk` as a sum of four terms:
        :math:`R_{ijk}^l =
        :math:`\partial_j \Gamma^l_{ki}`
        :math:`- \partial_k \Gamma^l_{ji}`
        :math:`+ \Gamma^l_{jm} \Gamma^m_{ki}`
        :math:`- \Gamma^l_{km} \Gamma^m_{ji}`

        Note that geomstats puts the contravariant index on
        the first dimension of the Christoffel symbols.

        Parameters
        ----------
        base_point :  array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        riemann_curvature : array-like, shape=[..., dim, dim,
                                                    dim, dim]
            riemann_tensor[...,i,j,k,l] = R_{ijk}^l
            Riemannian tensor curvature,
            with the contravariant index on the last dimension.
        """
        if len(self._space.shape) > 1:
            raise NotImplementedError(
                "Riemann tensor not implemented for manifolds with points of ndim > 1."
            )
        christoffels = self.christoffels(base_point)
        jacobian_christoffels = gs.autodiff.jacobian_vec(self.christoffels)(base_point)

        prod_christoffels = gs.einsum(
            "...ijk,...klm->...ijlm", christoffels, christoffels
        )
        riemann_curvature = (
            gs.einsum("...ijlm->...lmji", jacobian_christoffels)
            - gs.einsum("...ijlm->...ljmi", jacobian_christoffels)
            + gs.einsum("...ijlm->...mjli", prod_christoffels)
            - gs.einsum("...ijlm->...lmji", prod_christoffels)
        )

        return riemann_curvature

    def curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point):
        r"""Compute the Riemann curvature map R.

        For three tangent vectors at base point :math:`P`:
        - :math:`X|_P = tangent\_vec\_a`,
        - :math:`Y|_P = tangent\_vec\_b`,
        - :math:`Z|_P = tangent\_vec\_c`,
        the curvature(X, Y, Z, P) is defined by
        :math:`R(X,Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_[X, Y]Z`.

        The output is the tangent vector:
        :math:`dx^l(R(X, Y)Z) = R_{ijk}^l X_j Y_k Z_i`
        written with Einstein notation.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        base_point :  array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        curvature : array-like, shape=[..., dim]
            curvature(X, Y, Z, P)[..., l] = dx^l(R(X, Y)Z)
            Tangent vector at `base_point`.
        """
        riemann = self.riemann_tensor(base_point)
        curvature = gs.einsum(
            "...ijkl, ...j, ...k, ...i -> ...l",
            riemann,
            tangent_vec_a,
            tangent_vec_b,
            tangent_vec_c,
        )
        return curvature

    def ricci_tensor(self, base_point):
        r"""Compute Ricci curvature tensor at base_point.

        The Ricci curvature tensor :math:`\mathrm{Ric}_{ij}` is defined as:
        :math:`\mathrm{Ric}_{ij} = R_{ikj}^k`
        with Einstein notation.

        Parameters
        ----------
        base_point :  array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        ricci_tensor : array-like, shape=[..., dim, dim]
            ricci_tensor[...,i,j] = Ric_{ij}
            Ricci tensor curvature.
        """
        riemann_tensor = self.riemann_tensor(base_point)
        ricci_tensor = gs.einsum("...ijkj -> ...ik", riemann_tensor)
        return ricci_tensor

    def directional_curvature(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the directional curvature (tidal force operator).

        For two tangent vectors at base_point :math:`P`:
        - :math:`X|_P = tangent\_vec\_a`,
        - :math:`Y|_P = tangent\_vec\_b`,
        the directional curvature, better known
        in relativity as the tidal force operator,
        is defined by
        :math:`R_Y(X) = R(Y,X)Y`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        base_point :  array-like, shape=[..., dim]
            Base-point on the manifold.

        Returns
        -------
        directional_curvature : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        """
        return self.curvature(tangent_vec_b, tangent_vec_a, tangent_vec_b, base_point)

    def curvature_derivative(
        self,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        base_point=None,
    ):
        r"""Compute the covariant derivative of the curvature.

        For four tangent vectors at base_point :math:`P`:
        - :math:`H|_P = tangent\_vec\_a`,
        - :math:`X|_P = tangent\_vec\_b`,
        - :math:`Y|_P = tangent\_vec\_c`,
        - :math:`Z|_P = tangent\_vec\_d`,
        the covariant derivative of the curvature is defined as:
        :math:`(\nabla_H R)(X, Y) Z |_P`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        tangent_vec_d : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        base_point :  array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        curvature_derivative : array-like, shape=[..., dim]
            Tangent vector at base-point.
        """
        raise NotImplementedError(
            "The covariant derivative of the curvature is not implemented."
        )

    def directional_curvature_derivative(
        self, tangent_vec_a, tangent_vec_b, base_point=None
    ):
        r"""Compute the covariant derivative of the directional curvature.

        For tangent vector fields at base_point :math:`P`:
        - :math:`X|_P = tangent\_vec\_a`,
        - :math:`Y|_P = tangent\_vec\_b`,
        the covariant derivative (in the direction `X`)
        :math:`(\nabla_X R_Y)(X) |_P = (\nabla_X R)(Y, X) Y |_P` of the
        directional curvature (in the direction `Y`)
        :math:`R_Y(X) = R(Y, X) Y`
        is a quadratic tensor in `X` and `Y` that
        plays an important role in the computation of the moments of the
        empirical Fréchet mean.

        References
        ----------
        .. [Pennec] Pennec, Xavier. Curvature effects on the empirical mean in
            Riemannian and affine Manifolds: a non-asymptotic high
            concentration expansion in the small-sample regime. Preprint. 2019.
            https://arxiv.org/abs/1906.07418
        """
        return self.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_a, tangent_vec_b, base_point
        )

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
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
            represents the different initial conditions, and the second
            corresponds to time.
        """
        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end point or an initial tangent "
                "vector to define the geodesic."
            )
        if end_point is not None:
            if initial_tangent_vec is not None:
                raise ValueError(
                    "Cannot specify both an end point and an initial tangent vector."
                )

            # TODO: do it from exp otherwise
            _check_log_solver(self)
            return self.log_solver.geodesic_bvp(
                self._space,
                end_point,
                initial_point,
            )

        _check_exp_solver(self)
        return self.exp_solver.geodesic_ivp(
            self._space, initial_tangent_vec, initial_point
        )

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector
        along the geodesic between two points `base_point` and `end_point`
        or alternatively defined by :math:`t \mapsto exp_{(base\_point)}(
        t*direction)`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, [n, m]}]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., {dim, [n, m]}]
            Point on the manifold. Point to transport from.
        direction : array-like, shape=[..., {dim, [n, m]}]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default: None.
        end_point : array-like, shape=[..., {dim, [n, m]}]
            Point on the manifold. Point to transport to.
            Optional, default: None.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., {dim, [n, m]}]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        """
        raise NotImplementedError(
            "The closed-form solution of parallel transport is not known, "
            "use the ladder_parallel_transport instead."
        )

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image.

        Parameters
        ----------
        base_point : array-like, shape=[..., {dim, [n, m]}]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        raise NotImplementedError("The injectivity range is not implemented yet.")
