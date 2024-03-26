"""Common interface to hyperbolic spaces.

Lead author: Thomas Gerald.
"""

import geomstats.errors as errors
from geomstats.geometry._hyperbolic import HyperbolicDiffeo  # noqa
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.poincare_half_space import PoincareHalfSpace


class Hyperbolic:
    """Class for the n-dimensional Hyperbolic space.

    This class is a common interface to the different models of hyperbolic
    geometry:

    - the hyperboloid, embedded in Minkowski space of dimension dim + 1 as the set of
      points whose squared norm is equal to -1. This representation is called
      `extrinsic` here.
    - the Poincare ball, the open ball of the Euclidean space of dimension dim.
    - the Poincare half-space, the open space of points of the Euclidean
      space of  dimension dim, whose last coordinate is positive.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    coords_type : str, {'extrinsic', 'ball', 'half-space'}
        Default coordinates to represent points in hyperbolic space.
        Optional, default: 'extrinsic'.
    """

    def __new__(cls, dim, coords_type="extrinsic", equip=True):
        """Instantiate class that corresponds to the coords_type."""
        errors.check_parameter_accepted_values(
            coords_type,
            "coords_type",
            ["extrinsic", "ball", "half-space"],
        )
        if coords_type == "extrinsic":
            return Hyperboloid(dim, equip=equip)
        if coords_type == "ball":
            return PoincareBall(dim, equip=equip)
        return PoincareHalfSpace(dim, equip=equip)
