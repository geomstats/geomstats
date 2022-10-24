"""Matrix groups.

Lead author: Nina Miolane.
"""
import abc
import geomstats.backend as gs
from geomstats import matrices
from geomstats.spaces.core import Group


class MatrixGroup(Group, abc.ABC):

    def __init__(self, n, **kwargs):
        """
        n is the dimension of the regular group representation.
        """
        super().__init__(**kwargs)
        self.n = n

    @property
    def default_point_type(self):
        return "matrix"

    @property
    def identity(self):
        """Matrix identity."""
        return gs.eye(self.n)

    @staticmethod
    def compose(point_a, point_b):
        """Perform function composition corresponding to the Lie group.

        Multiply the elements `point_a` and `point_b`.

        Parameters
        ----------
        point_a : array-like, shape=[..., {dim, [n, n]}]
            Left factor in the product.
        point_b : array-like, shape=[..., {dim, [n, n]}]
            Right factor in the product.

        Returns
        -------
        composed : array-like, shape=[..., {dim, [n, n]}]
            Product of point_a and point_b along the first dimension.
        """
        return matrices.mul(point_a, point_b)

    @classmethod
    def inverse(cls, point):
        """Compute the inverse law of the Lie group.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
            Point to be inverted.

        Returns
        -------
        inverse : array-like, shape=[..., {dim, [n, n]}]
            Inverted point.
        """
        return gs.linalg.inv(point)

    @abc.abstractmethod
    def irrep(self, index):
        raise NotImplementedError

    def act(self, g, x):
        return matrices.mul(g, x)
