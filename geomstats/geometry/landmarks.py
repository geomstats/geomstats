"""Manifold for sets of landmarks that belong to any given manifold.

Lead author: Nicolas Guigui.
"""

import geomstats.backend as gs
from geomstats.integrator import integrate
from scipy.optimize import minimize
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.product_manifold import NFoldManifold
from geomstats.geometry.product_riemannian_metric import NFoldMetric
from geomstats.geometry.connection import N_STEPS

class Landmarks(NFoldManifold):
    """Class for space of landmarks.

    The landmark space is a product manifold where all manifolds in the
    product are the same. The default metric is the product metric and
    is often referred to as the L2 metric.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold to which landmarks belong.
    k_landmarks : int
        Number of landmarks.
    """

    def __init__(self, ambient_manifold, k_landmarks, **kwargs):
        kwargs.setdefault(
            "metric", L2LandmarksMetric(ambient_manifold.metric, k_landmarks)
        )
        super().__init__(base_manifold=ambient_manifold, n_copies=k_landmarks, **kwargs)
        self.ambient_manifold = ambient_manifold
        self.k_landmarks = k_landmarks


class L2LandmarksMetric(NFoldMetric):
    """L2 Riemannian metric on the space of landmarks.

    This is the NFoldMetric of the n-fold manifold made out
    of k_landmarks copies of the ambient manifold of each landmark.

    Parameters
    ----------
    ambient_metric : RiemannianMetric
        Riemannian metric of the manifold to which the landmarks belong.
    k_landmarks: int
        Number of landmarks.

    """

    def __init__(self, ambient_metric, k_landmarks, **kwargs):
        super().__init__(base_metric=ambient_metric, n_copies=k_landmarks, **kwargs)
        self.ambient_metric = ambient_metric
        self.k_landmarks = k_landmarks


class KernelLandmarksMetric(RiemannianMetric):
    r"""Kernel metric (in fact the kernel matrix gives the co-metric) for the LDDMM framework on landmark spaces.

    Parameters
    ----------
    ambient_dimension: int
        Dimension of the Euclidean space R^n containing the landmarks.
    k_landmarks : int
        Number of landmarks.
    kernel : callable
        Kernel function to generate the space of admissible vector fields. It
        should take two points of the ambient space as inputs and output a
        scalar. An example is the Gaussian kernel:
        .. math:

                    k(x, y) = exp(-|x-y|^2/ \sigma^2)
    """

    def __init__(
            self, ambient_dimension, k_landmarks,
            kernel=lambda d: gs.exp(-d)):
        super(KernelLandmarksMetric, self).__init__(
            dim=ambient_dimension * k_landmarks, shape=(k_landmarks, ambient_dimension))
        self.kernel = kernel
        self.ambient_dimension = ambient_dimension
        self.k_landmarks = k_landmarks

    def kernel_matrix(self, base_point):
        r"""Compute the kernel matrix, for a scalar kernel.

        .. math:
                    K_{i,j} = kernel(x_i, x_j)

        Where :math: `x_i` are the landmarks of the base point :math: `x`

        Parameters
        ----------
        base_point : landmark configuration :math: `x`

        Returns
        -------
        kernel_mat : [..., k_landmarks, k_landmarks]
        """
        squared_dist = gs.sum(
            (base_point[..., :, None, :] - base_point) ** 2, axis=-1)
        return self.kernel(squared_dist)
    
    def inner_coproduct(self, cotangent_vec_a, cotangent_vec_b, base_point):
        """Compute inner coproduct between two cotangent vectors at base point.

        This is the inner product associated to the cometric matrix.
        
        N.B. Here we override the default implementation to use the kernel matrix 
        instead of the cometric matric, for simplification.
        The kernel matrix is (n,n) where n is the number of landmarks,
        and the cometric matrix is (dim,dim)=(n*d,n*d) where d is the ambient dimension.
        The cometric matrix is just the Kronecker product between the kernel matrix and I_d
        (identity matrix of size d).

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
        return gs.einsum(
            "...ij,...ik,...kj", cotangent_vec_a, self.kernel_matrix(base_point), cotangent_vec_b
        )

    def tangent_to_cotangent(self, tangent_vec, base_point):
        r"""Converts tangent vector to cotangent vector
        """
        cotangent_vec = gs.einsum(
                "...ij,...j->...i", self.metric_matrix(base_point), tangent_vec
            )
        return cotangent_vec
    
    def cotangent_to_tangent(self, cotangent_vec, base_point):
        r"""Converts cotangent vector to tangent vector
        """
        tangent_vec = gs.einsum(
                "...ij,...j->...i", self.cometric_matrix(base_point), cotangent_vec
            )
        return tangent_vec

    def hamiltonian_equation(self, state, t):
        r"""Compute the right-hand-side of the Hamiltonian equations.

        Compute the partial derivatives of the Hamiltonian with respect to
        position and momentum by automatic differentiation. This gives a
        formulation of the geodesic equation:
        .. math:

                \dot_q = \partial_p H(q, p)
                \dot_p = -\partial_q H(q, p)

        Parameters
        ----------
        state : array-like, shape=[2, ..., dim]
            stack of position (Point on the manifold) and momentum (Covector at `position`).
        t : time index (unused, but needed for passing the function to the integrator)

        Returns
        -------
        array-like, shape=[2, ..., dim]
            stack of h_p (Partial derivative with respect to `position`) 
            and -h_q (Partial derivative with respect to `momentum`)
        """
        _, gradient = gs.autodiff.value_and_grad(self.hamiltonian)(state)
        h_q, h_p = gradient
        return gs.stack((h_p, - h_q))

    def exp(self, tangent_vec, base_point, n_steps=N_STEPS, step="euler", **kwargs):
        """Exponential map on the cotangent bundle.
        
        We redirect to exp_from_cotangent since the cometric is given instead of the metric.
        """
        cotangent_vec = self.tangent_to_cotangent(tangent_vec, base_point, n_steps, step, **kwargs)
        return self.exp_from_cotangent(cotangent_vec, base_point)

    def exp_from_cotangent(self, cotangent_vec, base_point, n_steps=N_STEPS, step='euler',
            point_type=None, **kwargs):
        """Exponential map on the cotangent bundle.

        Exponential map at base_point of cotangent_vec computed by integration
        of the geodesic equation (initial value problem), using the
        Hamiltonian equations.

        Parameters
        ----------
        cotangent_vec : array-like, shape=[..., dim]
            Cotangent vector at the base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_steps : int
            Number of discrete time steps to take in the integration.
            Optional, default: 10.
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
        flow = integrate(self.hamiltonian_equation, initial_state,
                            n_steps=n_steps, step=step)

        exp = flow[-1][0]
        return exp
    
    def log_as_cotangent(
        self,
        point,
        base_point,
        n_steps=N_STEPS,
        step="euler",
        max_iter=25,
        verbose=False,
        tol=gs.atol,
    ):
        """Compute logarithm map.

        Solve the boundary value problem associated to the hamiltonian equations
        and conjugate gradient descent.

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
        cotangent_vec : array-like, shape=[..., dim]
            Cotangent vector at the base point.
        """

        max_shape = point.shape
        if len(point.shape) <= len(base_point.shape):
            max_shape = base_point.shape

        def objective(momentum):
            """Define the objective function."""
            momentum = gs.array(momentum)
            momentum = gs.cast(momentum, dtype=base_point.dtype)
            momentum = gs.reshape(momentum, max_shape)
            delta = self.exp_from_cotangent(momentum, base_point, n_steps, step) - point
            return gs.sum(delta**2)

        objective_with_grad = gs.autodiff.value_and_grad(objective, to_numpy=True)

        cotangent_vec = gs.flatten(gs.random.rand(*max_shape))

        res = minimize(
            objective_with_grad,
            cotangent_vec,
            method="L-BFGS-B",
            jac=True,
            options={"disp": verbose, "maxiter": max_iter},
            tol=tol,
        )

        cotangent_vec = gs.array(res.x)
        cotangent_vec = gs.reshape(cotangent_vec, max_shape)
        cotangent_vec = gs.cast(cotangent_vec, dtype=base_point.dtype)

        return cotangent_vec
    
    def log(
        self,
        point,
        base_point,
        **kwargs
    ):
        """Compute logarithm map.
            Override default implementation to use the cometric instead of the metric
        """
        cotangent_vec = self.log_as_cotangent(point, base_point, **kwargs)
        tangent_vec = self.cotangent_to_tangent(cotangent_vec, base_point)
        return tangent_vec

    def geodesic(
        self, initial_point, end_point=None, initial_tangent_vec=None, **exp_kwargs
    ):
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
        point_type = self.default_point_type

        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end point or an initial tangent "
                "vector to define the geodesic."
            )
        if end_point is not None:
            shooting_cotangent_vec = self.log_as_cotangent(point=end_point, base_point=initial_point)
            if initial_tangent_vec is not None:
                shooting_tangent_vec = self.cotangent_to_tangent(shooting_cotangent_vec, initial_point)
                if not gs.allclose(shooting_tangent_vec, initial_tangent_vec):
                    raise RuntimeError(
                        "The shooting tangent vector is too"
                        " far from the input initial tangent vector."
                    )
            initial_cotangent_vec = shooting_cotangent_vec

        if point_type == "vector":
            if initial_point.ndim!=1 or initial_cotangent_vec.ndim!=1:
                raise ValueError(
                "batch mode not implemented here "
            )
        else:
            if initial_point.ndim!=2 or initial_cotangent_vec.ndim!=2:
                raise ValueError(
                "batch mode not implemented here "
            )

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
            if point_type == "vector":
                cotangent_vecs = gs.einsum("i,...k->...ik", t, initial_cotangent_vec)
            else:
                cotangent_vecs = gs.einsum("i,...kl->...ikl", t, initial_cotangent_vec)

            points_at_time_t = [
                self.exp_from_cotangent(ctv, initial_point, **exp_kwargs)
                for ctv in cotangent_vecs
            ]
            points_at_time_t = gs.stack(points_at_time_t, axis=0)

            return points_at_time_t

        return path