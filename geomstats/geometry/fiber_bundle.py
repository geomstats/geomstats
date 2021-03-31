"""Class for (principal) fiber bundles."""

from scipy.optimize import minimize

import geomstats.backend as gs
from geomstats.geometry.lie_group import LieGroup
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class FiberBundle(Manifold):
    """Class for (principal) fiber bundles.

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
    ambient_metric : RiemannianMetric
        Metric to use in the total space.
        Optional. The `metric` attribute of the total space is used if no
        ambient metric is passed.
    group_action : callable
        Right group action. It must take as input a point of the total space
        and an element of the group, and return a point of the total space.
    dim : int
        Dimension of the base manifold.
        Optional. If available the dimension of the base manifold is used,
        or the difference between the dimension of the total space and the
        group. Either dim, base or group must be given as input.
    """

    def __init__(
            self, total_space: Manifold, base: Manifold = None,
            group: LieGroup = None, ambient_metric: RiemannianMetric = None,
            group_action=None, dim: int = None, **kwargs):

        if dim is None:
            if base is not None:
                dim = base.dim
            elif group is not None:
                dim = total_space.dim - group.dim
            else:
                raise ValueError('Either the base manifold, '
                                 'its dimension, or the group acting on the '
                                 'total space must be provided.')

        super(FiberBundle, self).__init__(dim=dim, **kwargs)

        self.base = base
        self.total_space = total_space
        self.group = group
        self.ambient_metric = ambient_metric

        if group_action is None and group is not None:
            group_action = group.compose
        self.group_action = group_action

    def belongs(self, point, atol=1e-6):
        """Evaluate if a point belongs to the base manifold.

        Evaluate if a point belongs to the base manifold when it is given,
        otherwise to the total space.

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

    @staticmethod
    def submersion(point):
        """Project a point to base manifold.

        This is the projection of the fiber bundle, defined on the total
        space, with values in the base manifold. This map is surjective.
        By default, the base manifold  is not explicit but is identified with a
        local section of the fiber bundle, so the submersion is the identity
        map.

        Parameters
        ----------
        point : array-like, shape=[..., {ambient_dim, [n, n]}]
            Point of the total space.

        Returns
        -------
        projection : array-like, shape=[..., {dim, [n, n]}]
            Point of the base manifold.
        """
        return point

    @staticmethod
    def lift(point):
        """Lift a point to total space.

        This is a section of the fiber bundle, defined on the base manifold,
        with values in the total space. It means that submersion applied after
        lift results in the identity map. By default, the base manifold
        is not explicit but is identified with a section of the fiber bundle,
        so the lift is the identity map.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
            Point of the base manifold.

        Returns
        -------
        lift : array-like, shape=[..., {ambient_dim, [n, n]}]
            Point of the total space.
        """
        return point

    def tangent_submersion(self, tangent_vec, base_point):
        """Project a tangent vector to base manifold.

        This is the differential of the projection of the fiber bundle,
        defined on the tangent space of a point of the total space,
        with values in the tangent space of the projection of this point in the
        base manifold. This map is surjective. By default, the base manifold
        is not explicit but is identified with a horizontal section of the
        fiber bundle, so the tangent submersion is the horizontal projection.

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
        return self.horizontal_projection(tangent_vec, base_point)

    def align(self, point, base_point, max_iter=25, verbose=False, tol=1e-6):
        """Align point to base_point.

        Find the optimal group element g such that the base point and
        point.g are well positioned, meaning that the total space distance is
        minimized. This also means that the geodesic joining the base point
        and the aligned point is horizontal. By default, this is solved by a
        gradient descent in the Lie algebra.

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
        group = self.group
        initial_distance = self.ambient_metric.squared_dist(
            point, base_point)
        if isinstance(initial_distance, float) or initial_distance.shape == ():
            n_samples = 1
        else:
            n_samples = len(initial_distance)

        max_shape = (n_samples, group.dim) if n_samples > 1 else \
            (group.dim, )

        def wrap(param):
            """Wrap a parameter vector to a group element."""
            algebra_elt = gs.array(param)
            algebra_elt = gs.cast(algebra_elt, dtype=base_point.dtype)
            algebra_elt = group.lie_algebra.matrix_representation(
                algebra_elt)
            group_elt = group.exp(algebra_elt)
            return self.group_action(point, group_elt)

        objective_with_grad = gs.autograd.value_and_grad(
            lambda param: self.ambient_metric.squared_dist(
                wrap(param), base_point))

        tangent_vec = gs.flatten(gs.random.rand(*max_shape))
        res = minimize(
            objective_with_grad, tangent_vec, method='L-BFGS-B', jac=True,
            options={'disp': verbose, 'maxiter': max_iter}, tol=tol)

        return wrap(res.x)

    def horizontal_projection(self, tangent_vec, base_point):
        r"""Project to horizontal subspace.

        Compute the horizontal component of a tangent vector at a
        base point by removing the vertical component,
        or by computing a horizontal lift of the tangent projection.

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
        except (RecursionError, NotImplementedError):
            return self.horizontal_lift(
                self.tangent_submersion(tangent_vec, base_point),
                base_point)

    def vertical_projection(self, tangent_vec, base_point, **kwargs):
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
        vertical : array-like, shape=[..., {ambient_dim, [n, n]}]
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
            Optional, default: None.
        atol : float
            Absolute tolerance.
            Optional, default: 1e-6.

        Returns
        -------
        is_horizontal : bool
            Boolean denoting if tangent vector is horizontal.
        """
        return gs.all(gs.isclose(
            tangent_vec, self.horizontal_projection(tangent_vec, base_point),
            atol=atol), axis=(-2, -1))

    def is_vertical(self, tangent_vec, base_point, atol=1e-6):
        """Evaluate if the tangent vector is vertical at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector.
        base_point : array-like, shape=[..., {ambient_dim, [n, n]}]
            Point on the manifold.
            Optional, default: None.
        atol : float
            Absolute tolerance.
            Optional, default: 1e-6.

        Returns
        -------
        is_vertical : bool
            Boolean denoting if tangent vector is vertical.
        """
        return gs.all(gs.isclose(
            tangent_vec, self.vertical_projection(tangent_vec, base_point),
            atol=atol), axis=(-2, -1))

    def horizontal_lift(self, tangent_vec, point=None, base_point=None):
        """Lift a tangent vector to a horizontal vector in the total space.

        It means that horizontal lift is the inverse of the restriction of the
        tangent submersion to the horizontal space at point, where point must
        be in the fiber above the base point. By default, the base manifold
        is not explicit but is identified with a horizontal section of the
        fiber bundle, so the horizontal lift is the horizontal projection.

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
        if point is None:
            if base_point is not None:
                point = self.lift(base_point)
            else:
                raise ValueError('Either a point (of the total space) or a '
                                 'base point (of the base manifold) must be '
                                 'given.')
        return self.horizontal_projection(tangent_vec, point)

    def integrability_tensor(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the fundamental tensor A of the submersion.

        The fundamental tensor A is defined for tangent vectors of the total
        space by [O'Neill]_
        :math: `A_X Y = ver\nabla^M_{hor X} (hor Y)
            + hor \nabla^M_{hor X}( ver Y)`
        where :math: `hor,ver` are the horizontal and vertical projections.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., {ambient_dim, [n, n]}]
            Point of the total space.

        Returns
        -------
        vector : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector at `base_point`, result of the A tensor applied to
            `tangent_vec_a` and `tangent_vec_b`.

        References
        ----------
        [O'Neill]  O’Neill, Barrett. The Fundamental Equations of a Submersion,
        Michigan Mathematical Journal 13, no. 4 (December 1966): 459–69.
        https://doi.org/10.1307/mmj/1028999604.
        """
        raise NotImplementedError
