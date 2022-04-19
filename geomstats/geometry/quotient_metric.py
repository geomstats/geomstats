"""Classes for fiber bundles and quotient metrics.

Lead author: Nicolas Guigui.
"""

import geomstats.backend as gs
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

    def __init__(self, fiber_bundle: FiberBundle, dim: int = None, **kwargs):
        if dim is None:
            if fiber_bundle.base is not None:
                dim = fiber_bundle.base.dim
            elif fiber_bundle.group is not None:
                dim = fiber_bundle.dim - fiber_bundle.group.dim
            else:
                raise ValueError(
                    "Either the base manifold, "
                    "its dimension, or the group acting on the "
                    "total space must be provided."
                )
        super(QuotientMetric, self).__init__(
            dim=dim, default_point_type=fiber_bundle.default_point_type, **kwargs
        )

        self.fiber_bundle = fiber_bundle
        self.group = fiber_bundle.group
        self.ambient_metric = fiber_bundle.ambient_metric

    def inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point=None, fiber_point=None
    ):
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
                raise ValueError(
                    "Either a point (of the total space) or a "
                    "base point (of the quotient manifold) must "
                    "be given."
                )
        horizontal_a = self.fiber_bundle.horizontal_lift(
            tangent_vec_a, fiber_point=fiber_point
        )
        horizontal_b = self.fiber_bundle.horizontal_lift(
            tangent_vec_b, fiber_point=fiber_point
        )
        return self.ambient_metric.inner_product(
            horizontal_a, horizontal_b, fiber_point
        )

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
            tangent_vec, fiber_point=lift
        )
        return self.fiber_bundle.riemannian_submersion(
            self.ambient_metric.exp(horizontal_vec, lift)
        )

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
            self.ambient_metric.log(aligned, bp_fiber), bp_fiber
        )

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

    def curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point):
        r"""Compute the curvature.

        For three vectors fields :math:`X|_P = tangent_vec_a,
        Y|_P = tangent_vec_b, Z|_P = tangent_vec_c` with tangent vector
        specified in argument at the base point :math:`P`,
        the curvature is defined by :math:`R(X,Y)Z = \nabla_{[X,Y]}Z
        - \nabla_X\nabla_Y Z + \nabla_Y\nabla_X Z`.

        In the case of quotient metrics, the fundamental equations of a
        Riemannian submersion allow to compute the curvature of the base
        manifold from the one of the total space and a correction term that
        uses the integrability tensor A [O'Neill]_.

        In more details, let :math:`X, Y, Z` be the horizontal lift of
        vector fields extending the tangent vectors given in argument in a
        neighborhood of the base-point P in the base-space. Then the
        curvature of the base-space at the base-points is
        :math:`R(X,Y) Z = hor( R^T(X,Y) Z) - 2 A_Z A_X Y + A_X A_Y Z - A_Y
        A_X Z`, where :math:`R^T(X,Y)Z` is the curvature tensor of the
        total space.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at `base_point`.
        base_point :  array-like, shape=[..., {dim, [n, n]}]
            Point on the base manifold.

        Returns
        -------
        curvature : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at `base_point`.

        References
        ----------
        .. [O'Neill]  O’Neill, Barrett. The Fundamental Equations of a
        Submersion, Michigan Mathematical Journal 13, no. 4 (December 1966):
        459–69. https://doi.org/10.1307/mmj/1028999604.
        """
        bundle = self.fiber_bundle
        fiber_point = bundle.lift(base_point)
        horizontal_a = bundle.horizontal_lift(tangent_vec_a, base_point)
        horizontal_b = bundle.horizontal_lift(tangent_vec_b, base_point)
        horizontal_c = bundle.horizontal_lift(tangent_vec_c, base_point)

        top_curvature = self.ambient_metric.curvature(
            horizontal_a, horizontal_b, horizontal_c, fiber_point
        )
        projected_top_curvature = bundle.tangent_riemannian_submersion(
            top_curvature, fiber_point
        )

        f_ab = bundle.integrability_tensor(horizontal_a, horizontal_b, fiber_point)
        f_c_f_ab = bundle.integrability_tensor(horizontal_c, f_ab, fiber_point)
        f_c_f_ab = bundle.tangent_riemannian_submersion(f_c_f_ab, fiber_point)

        f_ac = bundle.integrability_tensor(horizontal_a, horizontal_c, fiber_point)
        f_b_f_ac = bundle.integrability_tensor(horizontal_b, f_ac, fiber_point)
        f_b_f_ac = bundle.tangent_riemannian_submersion(f_b_f_ac, fiber_point)

        f_bc = bundle.integrability_tensor(horizontal_b, horizontal_c, fiber_point)
        f_a_f_bc = bundle.integrability_tensor(horizontal_a, f_bc, fiber_point)
        f_a_f_bc = bundle.tangent_riemannian_submersion(f_a_f_bc, fiber_point)

        return projected_top_curvature - 2 * f_c_f_ab + f_a_f_bc - f_b_f_ac

    def curvature_derivative(
        self,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        base_point=None,
    ):
        r"""Compute the covariant derivative of the curvature.

        For four vectors fields :math:`H|_P = tangent_vec_a, X|_P =
        tangent_vec_b, Y|_P = tangent_vec_c, Z|_P = tangent_vec_d` with
        tangent vector value specified in argument at the base point `P`,
        the covariant derivative of the curvature
        :math:`(\nabla_H R)(X, Y)Z |_P` is computed at the base point P.

        In the case of quotient metrics, the fundamental equations of a
        Riemannian submersion allow to compute this tensor on the base manifold
        from the one of the total space T and its covariant derivative with
        additional correction terms involving the integrability tensor A and
        its covariant derivatives [Pennec]_.

        In more details, let :math:`H, X, Y, Z` be the horizontal lift of
        parallel vector fields extending the tangent vectors given in argument
        by parallel transport in a neighborhood of the base-point P in the
        base-space. Such vector fields verify :math:`\nabla^T_H H=0`,
        :math:`\nabla^T_H^X = A_H X` (and similarly for Y and Z) using the
        connection :math:`\nabla^T` of the total space. Then the covariant
        derivative of the curvature tensor is given by
        :math:`\nabla_H (R(X, Y) Z) =
        \hor\nabla_H^T(R^T(X,Y)Z) - A_H(ver R^T(X,Y)Z )
        + (2 A_H A_Z A_X Y - A_H A_X A_Y Z + A_H A_Y A_X Z)
        - (2 \nabla_H^T A_Z A_X Y - \nabla_H^T A_X A_Y Z +
             \nabla_H^T A_Y A_X Z)`, where :math:`R^T(X,Y)Z` is the curvature
        tensor of the total space.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at `base_point` (derivative direction)).
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_d : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., n, n]
            Point on the base manifold.

        Returns
        -------
        curvature_derivative : array-like, shape=[..., n, n]
            Tangent vector at base point.

        References
        ----------
        .. [Pennec] Pennec, Xavier. Computing the curvature and its gradient
        in Kendall shape spaces. Unpublished.
        """
        bundle = self.fiber_bundle
        point_fiber = bundle.lift(base_point)
        hor_h = bundle.horizontal_lift(tangent_vec_a, point_fiber)
        hor_x = bundle.horizontal_lift(tangent_vec_b, point_fiber)
        hor_y = bundle.horizontal_lift(tangent_vec_c, point_fiber)
        hor_z = bundle.horizontal_lift(tangent_vec_d, point_fiber)

        nabla_h_x = bundle.integrability_tensor(hor_h, hor_x, point_fiber)
        nabla_h_y = bundle.integrability_tensor(hor_h, hor_y, point_fiber)
        nabla_h_z = bundle.integrability_tensor(hor_h, hor_z, point_fiber)

        nabla_curvature_top = self.ambient_metric.curvature_derivative(
            hor_h, hor_x, hor_y, hor_z, point_fiber
        )

        hor_nabla_curvature_top = bundle.horizontal_projection(
            nabla_curvature_top, point_fiber
        )
        ver_nabla_curvature_top = nabla_curvature_top - hor_nabla_curvature_top

        a_h_ver_nabla_curvature_top = bundle.integrability_tensor(
            hor_h, ver_nabla_curvature_top, point_fiber
        )

        # A_H A_Z A_X Y and \nabla_H A_Z A_X Y
        nabla_h_a_x_y, a_x_y = bundle.integrability_tensor_derivative(
            hor_h, hor_x, nabla_h_x, hor_y, nabla_h_y, point_fiber
        )
        nabla_h_a_z_a_x_y, a_z_a_x_y = bundle.integrability_tensor_derivative(
            hor_h, hor_z, nabla_h_z, a_x_y, nabla_h_a_x_y, point_fiber
        )
        a_h_a_z_a_x_y = bundle.integrability_tensor(hor_h, a_z_a_x_y, point_fiber)

        # A_H A_X A_Y Z and \nabla_H A_X A_Y Z
        nabla_h_a_y_z, a_y_z = bundle.integrability_tensor_derivative(
            hor_h, hor_y, nabla_h_y, hor_z, nabla_h_z, point_fiber
        )
        nabla_h_a_x_a_y_z, a_x_a_y_z = bundle.integrability_tensor_derivative(
            hor_h, hor_x, nabla_h_x, a_y_z, nabla_h_a_y_z, point_fiber
        )
        a_h_a_x_a_y_z = bundle.integrability_tensor(hor_h, a_x_a_y_z, point_fiber)

        # A_H A_Y A_X Z and \nabla_H A_Y A_X Z
        nabla_h_a_x_z, a_x_z = bundle.integrability_tensor_derivative(
            hor_h, hor_x, nabla_h_x, hor_z, nabla_h_z, point_fiber
        )
        nabla_h_a_y_a_x_z, a_y_a_x_z = bundle.integrability_tensor_derivative(
            hor_h, hor_y, nabla_h_y, a_x_z, nabla_h_a_x_z, point_fiber
        )
        a_h_a_y_a_x_z = bundle.integrability_tensor(hor_h, a_y_a_x_z, point_fiber)

        return (
            hor_nabla_curvature_top
            - a_h_ver_nabla_curvature_top
            - 2.0 * (nabla_h_a_z_a_x_y - a_h_a_z_a_x_y)
            + (nabla_h_a_x_a_y_z - a_h_a_x_a_y_z)
            - (nabla_h_a_y_a_x_z - a_h_a_y_a_x_z)
        )

    def directional_curvature_derivative(
        self, tangent_vec_a, tangent_vec_b, base_point=None
    ):
        r"""Compute the covariant derivative of the directional curvature.

        For two vectors fields :math:`X|_P = tangent_vec_a, Y|_P =
        tangent_vec_b` with tangent vector value specified in argument at the
        base point `P`, the covariant derivative (in the direction 'X')
        :math:`(\nabla_X R_Y)(X) |_P = (\nabla_X R)(Y, X) Y |_P` of the
        directional curvature (in the direction `Y`)
        :math:`R_Y(X) = R(Y, X) Y`  is a quadratic tensor in 'X' and 'Y' that
        plays an important role in the computation of the moments of the
        empirical Fréchet mean.

        This tensor can be computed from the covariant derivative of the
        curvature tensor as in done generically the Connection class.
        However, in the case of quotient metrics, a simplified expression can
        be implemented based on the directional curvature of the total space T
        and its covariant derivative with additional correction terms
        involving the integrability tensor A and its covariant derivatives
        [Pennec]_.

        In more details, let :math:`X, Y` be the horizontal lift of parallel
        vector fields extending the tangent vectors given in argument by
        parallel transport in a neighborhood of the base-point P in the
        base-space. Such vector fields verify :math:`\nabla^T_X X=0` and
        :math:`\nabla^T_X^Y = A_X Y` using the connection :math:`\nabla^T`
        of the total space. Then the covariant derivative of the
        directional curvature tensor is given by :math:
        `\nabla_X (R_Y(X)) = hor \nabla^T_X (R^T_Y(X)) - A_X( ver R^T_Y(X))
        + 3 A_X A_Y A_X Y - 3 \nabla_X^T A_Y A_X Y `, where :math:`R^T_Y(X)`
        is the directional curvature tensor of the total space.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., n, n]
            Point on the base manifold.

        Returns
        -------
        curvature_derivative : array-like, shape=[..., n, n]
            Tangent vector at base point.

        References
        ----------
        .. [Pennec] Pennec, Xavier. Computing the curvature and its gradient
        in Kendall shape spaces. Unpublished.
        """
        bundle = self.fiber_bundle
        point_fiber = bundle.lift(base_point)
        hor_x = bundle.horizontal_lift(tangent_vec_a, point_fiber)
        hor_y = bundle.horizontal_lift(tangent_vec_b, point_fiber)

        nabla_x_x = gs.zeros_like(hor_x)
        nabla_x_y = bundle.integrability_tensor(hor_x, hor_y, point_fiber)

        nabla_curvature_top = self.ambient_metric.curvature_derivative(
            hor_x, hor_x, hor_y, hor_y, point_fiber
        )

        hor_nabla_curvature_top = bundle.horizontal_projection(
            nabla_curvature_top, point_fiber
        )
        ver_nabla_curvature_top = nabla_curvature_top - hor_nabla_curvature_top

        a_x_ver_nabla_curvature_top = bundle.integrability_tensor(
            hor_x, ver_nabla_curvature_top, point_fiber
        )

        # A_X A_Y A_X Y and \nabla_X A_Y A_X Y
        nabla_x_a_x_y, a_x_y = bundle.integrability_tensor_derivative(
            hor_x, hor_x, nabla_x_x, hor_y, nabla_x_y, point_fiber
        )
        nabla_x_a_y_a_x_y, a_y_a_x_y = bundle.integrability_tensor_derivative(
            hor_x, hor_y, nabla_x_y, a_x_y, nabla_x_a_x_y, point_fiber
        )
        a_x_a_y_a_x_y = bundle.integrability_tensor(hor_x, a_y_a_x_y, point_fiber)

        return (
            hor_nabla_curvature_top
            - a_x_ver_nabla_curvature_top
            + 3.0 * (nabla_x_a_y_a_x_y - a_x_a_y_a_x_y)
        )
