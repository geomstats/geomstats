"""Visualization for Geometric Statistics."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import geomstats.backend as gs

AX_SCALE = 1.2


class PoincarePolyDisk:
    """Class used to plot points in the Poincare polydisk."""

    def __init__(self, points=None, point_type="ball", n_disks=2):
        self.center = gs.array([0.0, 0.0])
        self.points = []
        self.point_type = point_type
        self.n_disks = n_disks
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None):
        """Define the ax parameters."""
        if ax is None:
            ax = plt.subplot()
        ax_s = AX_SCALE
        plt.setp(ax, xlim=(-ax_s, ax_s), ylim=(-ax_s, ax_s), xlabel="X", ylabel="Y")
        return ax

    def add_points(self, points):
        """Add points to draw."""
        if self.point_type == "extrinsic":
            points = self.convert_to_poincare_coordinates(points)
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def clear_points(self):
        """Clear the points to draw."""
        self.points = []

    @staticmethod
    def convert_to_poincare_coordinates(points):
        """Convert points to poincare coordinates."""
        poincare_coords = points[:, 1:] / (1 + points[:, :1])
        return poincare_coords

    def draw(self, ax, **kwargs):
        """Draw."""
        circle = plt.Circle((0, 0), radius=1.0, color="black", fill=False)
        ax.add_artist(circle)
        points_x = [gs.to_numpy(point[0]) for point in self.points]
        points_y = [gs.to_numpy(point[1]) for point in self.points]
        ax.scatter(points_x, points_y, **kwargs)
