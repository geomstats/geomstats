"""Bures-Wasserstein metric on SPD matrices as a quotient metric.

Importing this module registers the quotient structure of the general
linear group under the special orthogonal compose action, so that
`GeneralLinear.equip_with_quotient` induces the Bures-Wasserstein metric
on SPD matrices.
"""

import geomstats.backend as gs
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.group_action import SpecialOrthogonalComposeAction
from geomstats.geometry.manifold import register_quotient
from geomstats.geometry.matrices import Matrices, MatricesMetric


class GeneralLinearBuresWassersteinBundle(FiberBundle):
    """Fiber bundle from the general linear group to SPD matrices.

    Variant of :class:`geomstats.geometry.spd_matrices.SPDBuresWassersteinMetric`
    obtained as the quotient of the Frobenius metric on the general linear
    group by the right action of the special orthogonal group.

    Parameters
    ----------
    total_space : GeneralLinear
        Total space.
    """

    def __init__(self, total_space):
        super().__init__(total_space=total_space, aligner=True)

    @staticmethod
    def riemannian_submersion(point):
        return Matrices.mul(point, gs.transpose(point))

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        product = Matrices.mul(base_point, gs.transpose(tangent_vec))
        return 2 * Matrices.to_symmetric(product)

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        if base_point is None and fiber_point is None:
            raise ValueError(
                "Either a point (of the total space) or a "
                "base point (of the base manifold) must be "
                "given."
            )

        if base_point is None:
            base_point = self.riemannian_submersion(fiber_point)

        if fiber_point is None:
            fiber_point = self.lift(base_point)

        sylvester = gs.linalg.solve_sylvester(base_point, base_point, tangent_vec)
        return Matrices.mul(sylvester, fiber_point)

    @staticmethod
    def lift(point):
        return gs.linalg.cholesky(point)


register_quotient(
    Space=GeneralLinear,
    Metric=MatricesMetric,
    GroupAction=SpecialOrthogonalComposeAction,
    FiberBundle=GeneralLinearBuresWassersteinBundle,
)
