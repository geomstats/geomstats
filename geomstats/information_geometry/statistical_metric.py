"""Statistical metric to equip a statistical manifold."""

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric


class StatisticalMetric(RiemannianMetric):
    """
    Defines statistical metric and connection induced from a divergence.

    Uses definitions provided in Nielson's An Elementary Introduction to
    Information Geometry, Theorem 4 on page 15 (https://arxiv.org/abs/1808.08271)
    """

    def __init__(self, dim, divergence):
        self.dim = dim
        self.divergence = self._unpack_tensor(divergence)

    def _unpack_tensor(self, func):
        def wrapper(tensor):
            return func(tensor[..., : self.dim], tensor[..., self.dim :])

        return wrapper

    def metric_matrix(self, base_point):
        """Definition on line (51) on page 14."""
        hess = gs.autodiff.hessian(self.divergence)
        base_point_pair = gs.concatenate([base_point, base_point])
        return -1 * hess(base_point_pair)[: self.dim, self.dim :]

    def divergence_christoffels(self, base_point):
        """Definition on line (52) on page 14."""
        hess = gs.autodiff.hessian(self.divergence)
        jac_hess = gs.autodiff.jacobian(hess)
        base_point_pair = gs.concatenate([base_point, base_point])
        return -1 * jac_hess(base_point_pair)[:2, :2, 2:]

    def dual_divergence_christoffels(self, base_point):
        """Definition on line (53) on page 14."""
        hess = gs.autodiff.hessian(self.divergence)
        jac_hess = gs.autodiff.jacobian(hess)
        base_point_pair = gs.concatenate([base_point, base_point])
        return -1 * jac_hess(base_point_pair)[:2, 2:, 2:]

    def amari_divergence_tensor(self, base_point):
        """Definition on line (42) on page 12."""
        divergence_christoffels = self.divergence_christoffels(base_point)
        dual_divergence_christoffels = self.dual_divergence_christoffels(base_point)
        return dual_divergence_christoffels - divergence_christoffels
