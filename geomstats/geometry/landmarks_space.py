"""
Manifold, i.e. a topological space that locally resembles
Euclidean space near each point.
"""

import math
from geomstats.geometry.matrices_space import MatricesSpace


class Landmarks(MatricesSpace):
    """
    Class for landmarks.
    """

    def __init__(self, n_landmarks, ambient_dimension):
        """
        Parameters
        ---------
        n_landmarks : int
                      number of landmarks of all shapes

        ambient_dimension : int
                            dimension of the

        """

        assert isinstance(ambient_dimension, int)
        assert isinstance(n_landmarks, int)

        super(Landmarks, self).__init__(n_landmarks, d)

        self.dimension = n_landmarks * ambient_dimension
        self.n_landmarks = n_landmarks
        self.ambient_dimension = ambient_dimension
