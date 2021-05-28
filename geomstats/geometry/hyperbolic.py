"""Common interface to hyperbolic spaces."""

import geomstats.errors as errors
from geomstats.geometry._hyperbolic import _Hyperbolic
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.poincare_half_space import PoincareHalfSpace


class Hyperbolic(_Hyperbolic, Manifold):
    """Class for the n-dimensional Hyperbolic space.

    This class is a common interface to the different models of hyperbolic
    geometry:
    - the hyperboloid, embedded in Minkowski space of dimension dim + 1. This
    representation is called `extrinsic` here.
    - the Poincare ball, the open ball of the Euclidean space of dimension dim.
    - the Poincare half-space, the open space of points of the Euclidean
    space of  dimension dim, whose last coordinate is positive.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    default_coords_type : str, {'extrinsic', 'ball', 'half-space'}
        Default coordinates to represent points in hyperbolic space.
        Optional, default: 'extrinsic'.
    scale : int
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
        Optional, default: 1.
    """

    def __new__(cls, *args, default_coords_type='extrinsic', **kwargs):
        """Instantiate class that corresponds to the default_coords_type."""
        errors.check_parameter_accepted_values(
            default_coords_type, 'default_coords_type',
            ['extrinsic', 'ball', 'half-space'])
        if default_coords_type == 'extrinsic':
            return Hyperboloid(*args, **kwargs)
        if default_coords_type == 'ball':
            return PoincareBall(*args, **kwargs)
        return PoincareHalfSpace(*args, **kwargs)
