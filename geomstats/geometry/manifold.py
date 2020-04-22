"""Manifold module.

In other words, a topological space that locally resembles
Euclidean space near each point.
"""

import geomstats.error


class Manifold:
    """Class for manifolds."""

    def __init__(
            self, dim, default_point_type='vector',
            default_coords_type='intrinsic'):
        geomstats.error.check_integer(dim, 'dim')
        geomstats.error.check_parameter_accepted_values(
            default_point_type, 'default_point_type', ['vector', 'matrix'])

        self.dim = dim
        self.default_point_type = default_point_type
        self.default_coords_type = default_coords_type

    def belongs(point, point_type=None):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dim]
                 Input points.

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
        """
        raise NotImplementedError('belongs is not implemented.')

    def regularize(self, point, point_type=None):
        """Regularize a point to the canonical representation for the manifold.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dim]
                 Input points.

        Returns
        -------
        regularized_point : array-like, shape=[n_samples, dim]
        """
        regularized_point = point
        return regularized_point
