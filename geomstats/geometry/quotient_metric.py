"""Classes for fiber bundles and quotient metrics."""

from geomstats.geometry.fiber_bundle import FiberBundle
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

    def __init__(self, fiber_bundle: FiberBundle):
        super(QuotientMetric, self).__init__(
            dim=fiber_bundle.dim,
            default_point_type=fiber_bundle.default_point_type)

        self.fiber_bundle = fiber_bundle
        self.group = fiber_bundle.group
        self.ambient_metric = fiber_bundle.ambient_metric

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

    def curvature(
            self, tangent_vec_a, tangent_vec_b, tangent_vec_c,
            base_point):
        r"""Compute the curvature.

        For three tangent vectors at a base point :math: `X,Y,Z`,
        the curvature is defined by
        :math: `R(X,Y)Z = \nabla_{[X,Y]}Z
        - \nabla_X\nabla_Y Z + \nabla_Y\nabla_X Z`.

        In the case of quotient metrics, the fundamental equations of a
        Riemannian submersion allow to compute the curvature of the base
        manifold from the one of the total space and a correction term that
        uses the fundamental tensor A [O'Neill]_.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at `base_point`.
        base_point :  array-like, shape=[..., {dim, [n, n]}]
            Point on the group. Optional, default is the identity.

        Returns
        -------
        curvature : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at `base_point`.

        References
        ----------
        [O'Neill]  O’Neill, Barrett. The Fundamental Equations of a Submersion,
        Michigan Mathematical Journal 13, no. 4 (December 1966): 459–69.
        https://doi.org/10.1307/mmj/1028999604.
        """
        bundle = self.fiber_bundle
        point_fiber = bundle.lift(base_point)
        horizontal_a = bundle.horizontal_lift(
            tangent_vec_a, base_point)
        horizontal_b = bundle.horizontal_lift(
            tangent_vec_b, base_point)
        horizontal_c = bundle.horizontal_lift(
            tangent_vec_c, base_point)
        top_curvature = self.ambient_metric.curvature(
            horizontal_a, horizontal_b, horizontal_c, point_fiber)
        projected_top_curvature = bundle.tangent_submersion(
            top_curvature, point_fiber)

        f_ab = bundle.integrability_tensor(
            horizontal_a, horizontal_b, point_fiber)
        f_c_f_ab = bundle.integrability_tensor(
            horizontal_c, f_ab, point_fiber)
        f_c_f_ab = bundle.tangent_submersion(f_c_f_ab, point_fiber)

        f_ac = bundle.integrability_tensor(
            horizontal_a, horizontal_c, point_fiber)
        f_b_f_ac = bundle.integrability_tensor(
            horizontal_b, f_ac, point_fiber)
        f_b_f_ac = bundle.tangent_submersion(f_b_f_ac, point_fiber)

        f_bc = bundle.integrability_tensor(
            horizontal_b, horizontal_c, point_fiber)
        f_a_f_bc = bundle.integrability_tensor(
            horizontal_a, f_bc, point_fiber)
        f_a_f_bc = bundle.tangent_submersion(f_a_f_bc, point_fiber)

        return -(projected_top_curvature + 2 * f_c_f_ab - f_a_f_bc + f_b_f_ac)
