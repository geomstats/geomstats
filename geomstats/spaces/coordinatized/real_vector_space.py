import abc
import geomstats.backend as gs
from geomstats.spaces.core import VectorSpace


class RealVectorSpace(VectorSpace, abc.ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def default_point_type(self):
        """Point type.

        `vector` or `matrix`.
        """
        if len(self.shape) == 1:
            return "vector"
        return "matrix"

    def get_identity(self):
        """Get the identity of the group.

        Returns
        -------
        identity : array-like, shape=[n]
        """
        identity = gs.zeros(self.dim)
        return identity

    identity = property(get_identity)

    def _create_basis(self):
        """Create the canonical basis."""
        return gs.eye(self.dim)
