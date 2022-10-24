"""Abstract class for Lie algebras."""

import abc
import geomstats.backend as gs
from geomstats import matrices

class LieAlgebra(abc.ABC):

    def __init__(self, group, **kwargs):
        self._group = group

    @abc.abstractmethod
    def bracket(self):
        raise NotImplementedError

    def adjoint_representation(self):
        """
        u is an element of the Lie algebra
        """
        return lambda u: matrices.mul(self._group.jacobian_adjoint_representation(self._group.identity), u)
