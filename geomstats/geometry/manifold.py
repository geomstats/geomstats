"""Manifold module.

In other words, a topological space that locally resembles
Euclidean space near each point.
"""

import math


class Manifold(object):
    """Class for manifolds."""

    def __init__(self, dimension):

        if dimension:
            assert isinstance(dimension, int) or dimension == math.inf
            assert dimension > 0

        self.dimension = dimension

    def belongs(self, point, point_type=None):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        points : array-like, shape=[n_samples, dimension]
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
        points : array-like, shape=[n_samples, dimension]
                 Input points.

        Returns
        -------
        regularized_point : array-like, shape=[n_samples, dimension]
        """
        regularized_point = point
        return regularized_point
