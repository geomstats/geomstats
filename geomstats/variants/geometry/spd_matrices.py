"""Bures-Wasserstein metric on SPD matrices as a quotient metric.

Importing this module registers the quotient structure of the positive
general linear group under the special orthogonal compose action, so that
`PositiveGeneralLinear.equip_with_quotient` induces the Bures-Wasserstein
metric on SPD matrices.
"""

import geomstats.backend as gs
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import PositiveGeneralLinear
from geomstats.geometry.group_action import SpecialOrthogonalComposeAction
from geomstats.geometry.manifold import register_quotient
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.quotient_metric import QuotientMetric


class PositiveGeneralLinearGramBundle(FiberBundle):
    r"""Gram map bundle from the positive general linear group to SPD matrices.

    The Gram map :math:`X \mapsto XX^T` is invariant under the right action
    of the special orthogonal group, and induces a diffeomorphism from the
    quotient :math:`GL^+(n) / SO(n)` to SPD matrices.

    Variant of :class:`geomstats.geometry.spd_matrices.SPDBuresWassersteinMetric`
    obtained as the quotient of the Frobenius metric on the positive general
    linear group.

    Parameters
    ----------
    total_space : PositiveGeneralLinear
        Total space.
    """

    def __init__(self, total_space):
        super().__init__(total_space=total_space, aligner=True)

    @staticmethod
    def riemannian_submersion(point):
        r"""Project a point to SPD matrices with the Gram map.

        .. math::

            \pi(X) = XX^T

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point of the total space.

        Returns
        -------
        projection : array-like, shape=[..., n, n]
            SPD matrix.
        """
        return Matrices.mul(point, gs.transpose(point))

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        r"""Project a tangent vector to SPD matrices.

        Differential of the Gram map:

        .. math::

            d\pi_X(V) = XV^T + VX^T

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., n, n]
            Point of the total space.

        Returns
        -------
        projection : array-like, shape=[..., n, n]
            Symmetric matrix, tangent to SPD matrices at the projection
            of `base_point`.
        """
        product = Matrices.mul(base_point, gs.transpose(tangent_vec))
        return 2 * Matrices.to_symmetric(product)

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        r"""Lift a tangent vector to a horizontal vector in the total space.

        Denoting `tangent_vec` by :math:`W`, `base_point` by :math:`P`
        and `fiber_point` by :math:`X`, the horizontal lift is

        .. math::

            V = SX

        where :math:`S` is the symmetric solution of the Sylvester equation
        :math:`PS + SP = W`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector to SPD matrices at `base_point`.
        base_point : array-like, shape=[..., n, n]
            SPD matrix.
            Optional, default: None. In this case, `fiber_point` must be given,
            and `riemannian_submersion` is used to compute `base_point`.
        fiber_point : array-like, shape=[..., n, n]
            Point of the total space in the fiber above `base_point`.
            Optional, default: None. In this case, `lift` is used to compute
            `fiber_point`.

        Returns
        -------
        horizontal_lift : array-like, shape=[..., n, n]
            Horizontal tangent vector to the total space at `fiber_point`.
        """
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
        """Lift an SPD matrix to the total space with the Cholesky factor.

        The Cholesky factor is lower triangular with positive diagonal,
        so it has a positive determinant.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            SPD matrix.

        Returns
        -------
        lift : array-like, shape=[..., n, n]
            Point of the total space.
        """
        return gs.linalg.cholesky(point)


class QuotientSPDBuresWassersteinMetric(QuotientMetric):
    """Bures-Wasserstein metric on SPD matrices as a quotient metric.

    Variant of :class:`geomstats.geometry.spd_matrices.SPDBuresWassersteinMetric`
    obtained as the quotient of the Frobenius metric on the positive general
    linear group by the right action of the special orthogonal group.

    Parameters
    ----------
    space : SPDMatrices
        Base manifold.
    total_space : PositiveGeneralLinear
        Total space.
        Optional, default: None. In this case, it is equipped with the
        Frobenius metric and the special orthogonal compose action.
    """

    def __init__(self, space, total_space=None):
        if total_space is None:
            total_space = PositiveGeneralLinear(space.n, equip=False)
            total_space.equip_with_metric(MatricesMetric)

        if not hasattr(total_space, "group_action"):
            total_space.equip_with_group_action(
                SpecialOrthogonalComposeAction(total_space.n)
            )

        if not hasattr(total_space, "fiber_bundle"):
            total_space.equip_with_quotient()

        super().__init__(space=space, total_space=total_space)


register_quotient(
    Space=PositiveGeneralLinear,
    Metric=MatricesMetric,
    GroupAction=SpecialOrthogonalComposeAction,
    FiberBundle=PositiveGeneralLinearGramBundle,
)
