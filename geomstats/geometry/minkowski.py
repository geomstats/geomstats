"""Minkowski space.

Lead author: Nina Miolane.
"""

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.flat_riemannian_metric import FlatRiemannianMetric


class Minkowski(Euclidean):
    r"""Class for Minkowski space.

    This is the Euclidean space endowed with the inner-product of signature (
    dim-1, 1), i.e.
    .. math::
        ds^2 = - dx_1^2 + dx_2^2 + ... + dx_n^2

    Parameters
    ----------
    dim : int
       Dimension of Minkowski space.
    """

    def __new__(cls, dim, equip=True):
        """Instantiate a Minkowski space.

        This is an instance of the `Euclidean` class endowed with the
        `MinkowskiMetric`.
        """
        space = Euclidean(dim, equip=False)
        if equip:
            space.equip_with_metric(MinkowskiMetric)
        return space


class MinkowskiMetric(FlatRiemannianMetric):
    """Class for the pseudo-Riemannian Minkowski metric."""

    def __init__(self, space):
        signature = (space.dim - 1, 1)

        q, p = signature
        diagonal = gs.array([-1.0] * p + [1.0] * q)
        metric_mat = from_vector_to_diagonal_matrix(diagonal)

        super().__init__(space=space, metric_matrix=metric_mat, signature=signature)
