"""Visualization for Geometric Statistics."""
import geomstats._backend as gs
import matplotlib.pyplot as plt
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.visualization._plotting import Plotter
from mpl_toolkits.mplot3d import Axes3D  # NOQA


class KleinDisk(Plotter):
    """Class used to plot KleinDisk"""
    """Test comment for pull request update"""

    def __init__(self, points=None):
        super().__init__()

        self.center = gs.array([0.0, 0.0])
        self.points = []
        if points is not None:
            self.add_points(points)

        self._space = Hyperboloid(dim=2)
        self._metric = self._space.metric
        self._belongs = self.space.belongs

        self._ax_scale = 1.2
        self._dim = 2
        self._convert_points = self._convert_to_klein_coordinates

    @staticmethod
    def _convert_to_klein_coordinates(points):
        """Convert poincare coordinates to klein coordinates"""
        poincare_coords = points[:, 1:] / (1 + points[:, :1])
        poincare_radius = gs.linalg.norm(poincare_coords, axis=1)
        poincare_angle = gs.arctan2(poincare_coords[:, 1], poincare_coords[:, 0])

        klein_radius = 2 * poincare_radius / (1 + poincare_radius**2)
        klein_angle = poincare_angle

        coords_0 = gs.expand_dims(klein_radius * gs.cos(klein_angle), axis=1)
        coords_1 = gs.expand_dims(klein_radius * gs.sin(klein_angle), axis=1)
        klein_coords = gs.concatenate([coords_0, coords_1], axis=1)
        return klein_coords

    def plot_space(self, ax=None, color="black", **kwargs):
        ax = self.set_ax(ax=ax)

        circle = plt.Circle((0, 0), radius=1.0, color=color, fill=False)
        ax.add_artist(circle)
        points_x = [gs.to_numpy(point[0]) for point in self.points]
        points_y = [gs.to_numpy(point[1]) for point in self.points]
        ax.scatter(points_x, points_y, **kwargs)


class PoincareDisk(Plotter):
    def __init__(self, points=None, point_type="extrinsic"):
        super().__init__()
        self.center = gs.array([0.0, 0.0])
        # self.points = []
        # self.point_type = point_type
        # if points is not None:
        #    self.add_points(points)
        self._space = Hyperboloid(dim=2)
        self._metric = self._space.metric
        self._belongs = self.space.belongs

        self._ax_scale = 1.2
        self._convert_points = self._convert_to_poincare_coordinates

    @staticmethod
    def _convert_to_poincare_coordinates(points):
        poincare_coords = points[:, 1:] / (1 + points[:, :1])
        return poincare_coords

    def draw(self, ax, **kwargs):
        circle = plt.Circle((0, 0), radius=1.0, color="black", fill=False)
        ax.add_artist(circle)
        if len(self.points) > 0:
            if gs.all([len(point) == 2 for point in self.points]):
                points_x = gs.stack([point[0] for point in self.points], axis=0)
                points_y = gs.stack([point[1] for point in self.points], axis=0)
                ax.scatter(points_x, points_y, **kwargs)
            else:
                raise ValueError("Points do not have dimension 2.")

    def plot(self, points, ax=None, point_type=None, **point_draw_kwargs):
        if point_type is None:
            point_type = "extrinsic"
        poincare_disk = PoincareDisk(point_type=point_type)
        ax = poincare_disk.set_ax(ax=ax)
        self.points = []
        poincare_disk.add_points(points)
        poincare_disk.draw(ax, **point_draw_kwargs)
        plt.axis("off")


class PoincareHalfPlane(Plotter):
    """Class used to plot points in the Poincare Half Plane."""

    def __init__(self, points=None, point_type="half-space"):
        super().__init__()
        self.points = []
        self.point_type = point_type
        if points is not None:
            self.add_points(points)
        self._space = PoincareHalfSpace(dim=2)
        self._metric = self._space.metric
        self._belongs = self.space.belongs

        self._ax_scale = 1.2
        self._convert_points = self._convert_to_half_plane_coordinates

    @staticmethod
    def _convert_to_half_plane_coordinates(points):
        disk_coords = points[:, 1:] / (1 + points[:, :1])
        disk_x = disk_coords[:, 0]
        disk_y = disk_coords[:, 1]

        denominator = disk_x**2 + (1 - disk_y) ** 2
        coords_0 = gs.expand_dims(2 * disk_x / denominator, axis=1)
        coords_1 = gs.expand_dims((1 - disk_x**2 - disk_y**2) / denominator, axis=1)

        half_plane_coords = gs.concatenate([coords_0, coords_1], axis=1)
        return half_plane_coords

    def draw(self, ax, **kwargs):
        points_x = [gs.to_numpy(point[0]) for point in self.points]
        points_y = [gs.to_numpy(point[1]) for point in self.points]
        ax.scatter(points_x, points_y, **kwargs)

    def plot(self, points, ax=None, point_type=None, **point_draw_kwargs):
        if point_type is None:
            point_type = "half-space"
        poincare_half_plane = PoincareHalfPlane(point_type=point_type)
        # ax = poincare_half_plane.set_ax(ax=ax)
        ax = self.set_ax(ax=ax)
        self.points = []
        poincare_half_plane.add_points(points)
        poincare_half_plane.draw(ax, **point_draw_kwargs)
