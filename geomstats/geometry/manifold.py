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

    def belongs(self, point):
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


    def is_tangent(self, vector, base_point=None):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dim_embedding]
            Vector.
        base_point : array-like, shape=[n_samples, dim_embedding]
            Point on the manifold.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        raise NotImplementedError(
            'is_tangent is not implemented.')

    def regularize(self, point):
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
