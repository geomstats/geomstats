"""Visualization for Klein Bottle."""

import geomstats._backend as gs
import matplotlib.pyplot as plt
from geomstats.geometry.klein_bottle import KleinBottle
from mpl_toolkits.mplot3d import Axes3D  # NOQA

AX_SCALE = 1.2

KB2 = KleinBottle(dim=2, equip=True)


class KleinBottle2D:
    """Class used to plot points on the two dimensional Klein Bottle."""

    def __init__(self, points=None, coords_type="intrinsic"):
        """Initialize points on manifold."""
        self.points = []
        if points is not None:
            self.add_points(points)

    def set_ax(ax=None):
        """Set the axis of the figure."""
        if ax is None:
            ax = plt.subplot()
        plt.setp(ax, xlabel="X", ylabel="Y")
        return ax

    def add_points(self, points):
        """Add points to the manifold."""
        if not gs.all(KB2.belongs(points)):
            raise ValueError("Points do not belong to Klein Bottle.")
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def draw_points(self, ax, points=None, **kwargs):
        """Draw 2D scatter or point cloud on the figure."""
        if points is None:
            points = self.points
        points_x = [gs.to_numpy(point[0]) for point in self.points]
        points_y = [gs.to_numpy(point[1]) for point in self.points]
        ax.scatter(points_x, points_y, **kwargs)

    def plot(self, points, ax, coords_type, **point_draw_kwargs):
        """Plot Klein Bottle visualization."""
        if coords_type != "intrinsic":
            coords_type = "intrinsic"
        kb = KleinBottle(coords_type=coords_type)
        ax = kb.set_ax(ax=ax)
        self.points = []
        kb.add_points(points)
        kb.draw(ax, **point_draw_kwargs)

    def clear_points(self):
        """Clear the points to draw."""
        self.points = []
