"""Classes for fiber bundles and quotient metrics."""

from geomstats.geometry.riemannian_metric import RiemannianMetric


class QuotientMetric(RiemannianMetric):
    """Quotient metric.

    Given a (principal) fiber bundle, or more generally a manifold with a
    Lie group acting on it by the right, the quotient space is the space of
    orbits under this action. The quotient metric is defined such that the
    canonical projection is a Riemannian submersion, i.e. it is isometric to
    the restriction of the metric of the total space to horizontal subspaces.

    Parameters
    ----------
    fiber_bundle : geomstats.geometry.fiber_bundle.FiberBundle
        Bundle structure to define the quotient.
    group : LieGroup
        Group acting on the right.
        Optional, default : None. In this case the group must be passed to
        the fiber bundle instance.
    ambient_metric : RiemannianMetric
        Metric of the total space.
        Optional, default : None. In this case, the total space must have a
        metric as an attribute.
    """

    def __init__(self, fiber_bundle, group=None, ambient_metric=None):
        super(QuotientMetric, self).__init__(
            dim=fiber_bundle.dim,
            default_point_type=fiber_bundle.default_point_type)

        self.fiber_bundle = fiber_bundle
        if group is None:
            group = fiber_bundle.group
        if ambient_metric is None:
            ambient_metric = fiber_bundle.total_space.metric

        self.group = group
        self.ambient_metric = ambient_metric

    def inner_product(
            self, tangent_vec_a, tangent_vec_b, base_point=None, point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector to the quotient manifold.
        tangent_vec_b : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector to the quotient manifold.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point on the quotient manifold.
            Optional, default: None.
        point : array-like, shape=[..., {dim, [n, n]}]
            Point on the total space, lift of `base_point`, i.e. such that
            `submersion` applied to `point` results in `base_point`.
            Optional, default: None. In this case, it is computed using the
            method lift.

        Returns
        -------
        inner_product : float, shape=[...]
            Inner products
        """
        if point is None:
            if base_point is not None:
                point = self.fiber_bundle.lift(base_point)
            else:
                raise ValueError('Either a point (of the total space) or a '
                                 'base point (of the quotient manifold) must '
                                 'be given.')
        horizontal_a = self.fiber_bundle.horizontal_lift(tangent_vec_a, point)
        horizontal_b = self.fiber_bundle.horizontal_lift(tangent_vec_b, point)
        return self.ambient_metric.inner_product(
            horizontal_a, horizontal_b, point)

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector to the quotient manifold.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point on the quotient manifold.
            Optional, default: None.

        Returns
        -------
        exp : array-like, shape=[..., {dim, [n, n]}]
            Point on the quotient manifold.
        """
        lift = self.fiber_bundle.lift(base_point)
        horizontal_vec = self.fiber_bundle.horizontal_lift(
            tangent_vec, lift)
        return self.fiber_bundle.submersion(
            self.ambient_metric.exp(horizontal_vec, lift))

    def log(self, point, base_point, **kwargs):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
            Point on the quotient manifold.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point on the quotient manifold.

        Returns
        -------
        log : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        point_fiber = self.fiber_bundle.lift(point)
        bp_fiber = self.fiber_bundle.lift(base_point)
        aligned = self.fiber_bundle.align(point_fiber, bp_fiber, **kwargs)
        return self.fiber_bundle.tangent_submersion(
            self.ambient_metric.log(aligned, bp_fiber), bp_fiber)

    def squared_dist(self, point_a, point_b, **kwargs):
        """Squared geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[...,  {dim, [n, n]}]
            Point.
        point_b : array-like, shape=[...,  {dim, [n, n]}]
            Point.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
            Squared distance.
        """
        lift_a = self.fiber_bundle.lift(point_a)
        lift_b = self.fiber_bundle.lift(point_b)
        aligned = self.fiber_bundle.align(lift_a, lift_b, **kwargs)
        return self.ambient_metric.squared_dist(aligned, lift_b)
