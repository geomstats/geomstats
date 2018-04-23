"""Base class for matrix Lie groups."""

import numpy as np

from geomstats.embedded_manifold import EmbeddedManifold
from geomstats.general_linear_group import GeneralLinearGroup
from geomstats.lie_group import LieGroup


class MatrixLieGroup(LieGroup, EmbeddedManifold):
    """
    Base class for matrix Lie groups,
    where each element is represented by a matrix by default.

    Note: for now, SO(n) and SE(n) elements are represented
    by a vector by default.
    """

    def __init__(self, dimension, n):
        LieGroup.__init__(
            self,
            dimension=dimension,
            identity=np.eye(n))
        EmbeddedManifold.__init__(
            self,
            dimension=dimension,
            embedding_manifold=GeneralLinearGroup(n=n))
