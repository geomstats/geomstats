"""Classes for quotient manifolds and quotient metrics"""

from scipy.optimize import minimize

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class FiberBundle(Manifold):
    """Class for (principal) fiber bundles

    This class implements abstract methods for fiber bundles, or more
    generally manifolds, with a submersion map, or a right Lie group action.

    Parameters
    ----------
    total_space : Manifold
        Total space of the bundle.
    base : Manifold
        Base manifold of the bundle.
        Optional. Default : None.
    group : LieGroup
        Group that acts on the total space by the right.
        Optional. Default : None.
        Either the group or the group action must be given.
    group_action : callable
        Right group action. It must take as input a point of the total space
        and an element of the group, and return a point of the total space.
    dim : int
        Dimension of the base manifold.
        Optional. If available the dimension of the base manifold is used,
        or the difference between the dimension of the total space and the
        group. Either dim, base or group must be given as input.
    """

    def __init__(self, total_space, base=None, group=None, group_action=None,
                 dim=None):

        if dim is None:
            if base is not None:
                dim = base.dim
            elif group is not None:
                dim = total_space.dim - group.dim
            else:
                raise ValueError('Either the base manifold, '
                                 'its dimension, or the group acting on the '
                                 'total space must be provided.')

        super(FiberBundle, self).__init__(
            dim=dim, default_point_type=total_space.default_point_type)

        self.base = base
        self.total_space = total_space
        self.group = group

        if group_action is None and group is not None:
            group_action = group.compose
        self.group_action = group_action

    def belongs(self, point, atol=1e-6):
        """Evaluate if a point belongs to the bundle.

        Evaluate if a point belongs to the base manifold is given, otherwise
        to the total space.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: 1e-6.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        if self.base is not None:
            return self.base.belongs(point, atol=atol)
        return self.total_space.belongs(point, atol=atol)

    def submersion(self, point):
        """Project a point to base manifold.

        This is the projection of the fiber bundle, defined on the total
        space, with values in the base manifold. This map is surjective.

        Parameters
        ----------
        point: array-like, shape=[..., {ambient_dim, [n, n]}]
            Point of the total space.

        Returns
        -------
        projection: array-like, shape=[..., {dim, [n, n]}]
            Point of the base manifold.
        """
        raise NotImplementedError

    def lift(self, point):
        """Lift a point to total space.

        This is a section of the fiber bundle, defined on the base manifold,
        with values in the total space. It means that lift composed with
        submersion results in the identity map.

        Parameters
        ----------
        point: array-like, shape=[..., {dim, [n, n]}]
            Point of the base manifold.

        Returns
        -------
        lift: array-like, shape=[..., {ambient_dim, [n, n]}]
            Point of the total space.
        """
        raise NotImplementedError

    def tangent_submersion(self, tangent_vec, base_point):
        """Project a tangent vector to base manifold.

        This is the differential of the projection of the fiber bundle,
        defined on the tangent space of a point of the total space,
        with values in the tangent space of the projection of this point in the
        base manifold. This map is surjective.

        Parameters
        ----------
        tangent_vec :  array-like, shape=[..., ambient_dim, [n , n]}]
            Tangent vector to the total space at `base_point`.
        base_point: array-like, shape=[..., {ambient_dim, [n, n]}]
            Point of the total space.

        Returns
        -------
        projection: array-like, shape=[..., {dim, [n, n]}]
            Tangent vector to the base manifold.
        """
        raise NotImplementedError

    def align(self, point, base_point, max_iter=25, verbose=False, tol=1e-6):
        """Align point to base_point.

        Find the optimal group element g such that the base point and
        point.g are well positioned, meaning that the total space distance is
        minimized. By default, this is solved by gradient descent.

        Parameters
        ----------
        point : array-like, shape=[..., {ambient_dim, [n, n]}]
            Point on the manifold.
        base_point : array-like, shape=[..., {ambient_dim, [n, n]}]
            Point on the manifold.
        max_iter : int
            Maximum number of gradient steps.
            Optional, default : 25.
        verbose : bool
            Verbosity level.
            Optional, default : False.
        tol : float
            Tolerance for the stopping criterion.
            Optional, default : 1e-6

        Returns
        -------
        aligned : array-like, shape=[..., {ambient_dim, [n, n]}]
            Action of the optimal g on point.
        """
        identity = self.group.identity
        initial_distance = self.total_space.metric.squared_dist(
            point, base_point)
        n_samples = 1 if isinstance(initial_distance, float) else \
            len(initial_distance)
        max_shape = (n_samples, ) + identity.shape if n_samples > 1 else \
            identity.shape

        def objective(param):
            group_elt = gs.array(param, dtype=base_point.dtype)
            group_elt = gs.reshape(group_elt, max_shape)
            aligned = self.group_action(param, group_elt)
            return self.total_space.metric.squared_dist(aligned, base_point)

        objective_with_grad = gs.autograd.value_and_grad(objective)
        tangent_vec = gs.flatten(gs.random.rand(*max_shape))
        res = minimize(
            objective_with_grad, tangent_vec, method='L-BFGS-B', jac=True,
            options={'disp': verbose, 'maxiter': max_iter}, tol=tol)

        optimal_group_elt = gs.array(res.x)
        optimal_group_elt = gs.reshape(
            optimal_group_elt, max_shape)
        return self.group_action(optimal_group_elt)(point)

    def horizontal_projection(self, tangent_vec, base_point):
        r"""Project to horizontal subspace.

        Compute the horizontal component of a tangent vector :math: `w` at a
        base point :math: `x` by removing the vertical component,
        or by a computing a horizontal lift of the tangent projection.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., {ambient_dim, [n, n]}]
            Point on the total space.

        Returns
        -------
        horizontal : array-like, shape=[..., {ambient_dim, [n, n]}]
            Horizontal component of `tangent_vec`.
        """
        try:
            return tangent_vec - self.vertical_projection(
                tangent_vec, base_point)
        except RecursionError:
            return self.horizontal_lift(
                self.tangent_submersion(tangent_vec, base_point), base_point)

    def vertical_projection(self, tangent_vec, base_point):
        r"""Project to vertical subspace.

        Compute the vertical component of a tangent vector :math: `w` at a
        base point :math: `x` by removing the horizontal component.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., {ambient_dim, [n, n]}]
            Point on the total space.

        Returns
        -------
        horizontal : array-like, shape=[..., {ambient_dim, [n, n]}]
            Vertical component of `tangent_vec`.
        """
        try:
            return tangent_vec - self.horizontal_projection(
                tangent_vec, base_point)
        except RecursionError:
            raise NotImplementedError

    def is_horizontal(self, tangent_vec, base_point, atol=1e-6):
        """Evaluate if the tangent vector is horizontal at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector.
        base_point : array-like, shape=[..., {ambient_dim, [n, n]}]
            Point on the manifold.
            Optional, default: none.
        atol : float
            Absolute tolerance.
            Optional, default: 1e-6.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if tangent vector is horizontal.
        """
        return gs.isclose(
            tangent_vec, self.horizontal_projection(tangent_vec, base_point),
            atol=atol)

    def is_vertical(self, tangent_vec, base_point, atol=1e-6):
        """Evaluate if the tangent vector is vertical at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector.
        base_point : array-like, shape=[..., {ambient_dim, [n, n]}]
            Point on the manifold.
            Optional, default: none.
        atol : float
            Absolute tolerance.
            Optional, default: 1e-6.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if tangent vector is vertical.
        """
        return gs.isclose(
            tangent_vec, self.vertical_projection(tangent_vec, base_point))

    def horizontal_lift(self, tangent_vec, point=None, base_point=None):
        """Lift a tangent vector to a horizontal vector in the total space.

        It means that horizontal lift is the inverse of the restriction of the
        tangent submersion to the horizontal space a point, where point must
        be in the fiber above the base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
        point: array-like, shape=[..., {ambient_dim, [n, n]}]
            Point of the total space.
            Optional, default : None. The `lift` method is used to compute a
            point at which to compute a tangent vector.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point of the base space.
            Optional, default : None. In this case, point must be given,
            and `submersion` is used to compute the base_point if needed.

        Returns
        -------
        horizontal_lift : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector to the total space at point.
        """
        raise NotImplementedError


class QuotientMetric(RiemannianMetric):
    def __init__(self, fiber_bundle, group, ambient_metric):
        super(QuotientMetric, self).__init__(
            dim=fiber_bundle.dim - group.dim,
            default_point_type=fiber_bundle.default_point_type)

        self.fiber_bundle = fiber_bundle
        self.group = group
        self.ambient_metric = ambient_metric

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        horizontal_a = self.fiber_bundle.horizontal_projection(
            tangent_vec_a, base_point)
        horizontal_b = self.fiber_bundle.horizontal_projection(
            tangent_vec_b, base_point)
        return self.ambient_metric.inner_product(
            horizontal_a, horizontal_b, base_point)

    def exp(self, tangent_vec, base_point, **kwargs):
        lift = self.fiber_bundle.lift(base_point)
        horizontal_vec = self.fiber_bundle.horizontal_projection(
            tangent_vec, lift)
        return self.fiber_bundle.submersion(
            self.ambient_metric.exp(horizontal_vec, lift))

    def log(self, point, base_point, **kwargs):
        aligned = self.fiber_bundle.align(point, base_point, **kwargs)
        return self.fiber_bundle.tangent_submersion(
            self.ambient_metric.log(aligned, base_point), base_point)

    def squared_dist(self, point_a, point_b, **kwargs):
        aligned = self.fiber_bundle.align(point_a, point_b, **kwargs)
        return self.ambient_metric.squared_dist(aligned, point_b)
