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
    """

    def __init__(self, fiber_bundle: FiberBundle, dim: int = None):
        if dim is None:
            if fiber_bundle.base is not None:
                dim = fiber_bundle.base.dim
            elif fiber_bundle.group is not None:
                dim = fiber_bundle.dim - fiber_bundle.group.dim
            else:
                raise ValueError('Either the base manifold, '
                                 'its dimension, or the group acting on the '
                                 'total space must be provided.')
        super(QuotientMetric, self).__init__(
            dim=dim,
            default_point_type=fiber_bundle.default_point_type)

        self.fiber_bundle = fiber_bundle
        self.group = fiber_bundle.group
        self.ambient_metric = fiber_bundle.ambient_metric

    def inner_product(
            self, tangent_vec_a, tangent_vec_b, base_point=None,
            fiber_point=None):
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
        fiber_point : array-like, shape=[..., {dim, [n, n]}]
            Point on the total space, lift of `base_point`, i.e. such that
            `submersion` applied to `point` results in `base_point`.
            Optional, default: None. In this case, it is computed using the
            method lift.

        Returns
        -------
        inner_product : float, shape=[...]
            Inner products
        """
        if fiber_point is None:
            if base_point is not None:
                fiber_point = self.fiber_bundle.lift(base_point)
            else:
                raise ValueError('Either a point (of the total space) or a '
                                 'base point (of the quotient manifold) must '
                                 'be given.')
        horizontal_a = self.fiber_bundle.horizontal_lift(
            tangent_vec_a, fiber_point=fiber_point)
        horizontal_b = self.fiber_bundle.horizontal_lift(
            tangent_vec_b, fiber_point=fiber_point)
        return self.ambient_metric.inner_product(
            horizontal_a, horizontal_b, fiber_point)

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
            tangent_vec, fiber_point=lift)
        return self.fiber_bundle.riemannian_submersion(
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
        fiber_point = self.fiber_bundle.lift(point)
        bp_fiber = self.fiber_bundle.lift(base_point)
        aligned = self.fiber_bundle.align(fiber_point, bp_fiber, **kwargs)
        return self.fiber_bundle.tangent_riemannian_submersion(
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
        fiber_point = bundle.lift(base_point)
        horizontal_a = bundle.horizontal_lift(tangent_vec_a, base_point)
        horizontal_b = bundle.horizontal_lift(tangent_vec_b, base_point)
        horizontal_c = bundle.horizontal_lift(tangent_vec_c, base_point)

        top_curvature = self.ambient_metric.curvature(
            horizontal_a, horizontal_b, horizontal_c, fiber_point)
        projected_top_curvature = bundle.tangent_riemannian_submersion(
            top_curvature, fiber_point)

        f_ab = bundle.integrability_tensor(
            horizontal_a, horizontal_b, fiber_point)
        f_c_f_ab = bundle.integrability_tensor(
            horizontal_c, f_ab, fiber_point)
        f_c_f_ab = bundle.tangent_riemannian_submersion(f_c_f_ab, fiber_point)

        f_ac = bundle.integrability_tensor(
            horizontal_a, horizontal_c, fiber_point)
        f_b_f_ac = bundle.integrability_tensor(
            horizontal_b, f_ac, fiber_point)
        f_b_f_ac = bundle.tangent_riemannian_submersion(f_b_f_ac, fiber_point)

        f_bc = bundle.integrability_tensor(
            horizontal_b, horizontal_c, fiber_point)
        f_a_f_bc = bundle.integrability_tensor(
            horizontal_a, f_bc, fiber_point)
        f_a_f_bc = bundle.tangent_riemannian_submersion(f_a_f_bc, fiber_point)

        return projected_top_curvature - 2 * f_c_f_ab + f_a_f_bc - f_b_f_ac
