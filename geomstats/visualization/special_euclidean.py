"""Visualization for Geometric Statistics."""
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.visualization._plotting import Plotter
from mpl_toolkits.mplot3d import Axes3D  # NOQA


class SpecialEuclidean2(Plotter):
    """Create the points point_a and point_b to plot a special euclidean space.

    Create point_a and point_b to create the wire frame of a special euclidean space.
    Their point_type is matrix.
    """

    def __init__(self, point_type="matrix"):
        super().__init__()

        self.point_type = point_type
        self._space = SpecialEuclidean(point_a=[..., 2], point_b=[..., 2])
        self._metric = SpecialEuclidean(point_a=[..., 2], point_b=[..., 2])
        self._belongs = self._space.belongs
