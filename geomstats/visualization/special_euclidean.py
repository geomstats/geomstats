"""Visualization for Geometric Statistics."""

import logging

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean

SE2_MAT = SpecialEuclidean(n=2, point_type="matrix")
SE2_VECT = SpecialEuclidean(n=2, point_type="vector")

SE3_MAT = SpecialEuclidean(n=3, point_type="matrix")
SE3_VECT = SpecialEuclidean(n=3, point_type="vector")

METRIC_SE2 = SE2_VECT.left_canonical_metric
METRIC_SE3 = SE3_VECT.left_canonical_metric


class SpecialEuclidean2:
    """Class used to plot points in the 2d special euclidean group."""

    def __init__(self, points=None, point_type="matrix"):
        """Initialize SE(2) objects."""
        self.points = []
        self.point_type = point_type
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None, x_lim=None, y_lim=None):
        """Define SE(2) plot axes."""
        if ax is None:
            ax = plt.subplot()
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        return ax

    def add_points(self, points):
        """Add points to SE(2) object."""
        if self.point_type == "vector":
            points = SE2_VECT.matrix_from_vector(points)
        if not gs.all(SE2_MAT.belongs(points)):
            logging.warning("Some points do not belong to SE2.")
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def draw_points(self, ax, points=None, **kwargs):
        """Visualization for SE(2) points."""
        if points is None:
            points = gs.stack(self.points)
        translation = points[..., :2, 2]
        frame_1 = points[:, :2, 0]
        frame_2 = points[:, :2, 1]
        ax.quiver(
            translation[:, 0],
            translation[:, 1],
            frame_1[:, 0],
            frame_1[:, 1],
            width=0.005,
            color="b",
        )
        ax.quiver(
            translation[:, 0],
            translation[:, 1],
            frame_2[:, 0],
            frame_2[:, 1],
            width=0.005,
            color="r",
        )
        ax.scatter(translation[:, 0], translation[:, 1], s=16, **kwargs)

    @staticmethod
    def plot_geodesic(
        initial_point, initial_tangent_vec, metric=METRIC_SE2, n_steps=40
    ):
        """Plot geodesic of SE(2).

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Point for the geodesic source.
        initial_tangent_vec : array-like, shape=[..., dim]
            Vector for the geodesic function.
        metric
            Chosen metric for geodesic
        n_steps : array-like, shape=[..., dim]
            Number of samples on the geodesic to plot.
        """
        geodesic = metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )
        t = gs.linspace(-3.0, 3.0, n_steps)
        points = geodesic(t)
        points_mat = SE2_VECT.matrix_from_vector(
            points
        )  # required as plot for SpecialEuclidean2 expects matrix form
        visualization.plot(points_mat, space="SE2_GROUP")


class SpecialEuclidean3:
    """Class used to plot points in the 3d special euclidean group."""

    def __init__(self, points=None, point_type="matrix"):
        """Initialize SE(3) objects."""
        self.points = []
        self.point_type = point_type
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None, x_lim=None, y_lim=None):
        """Define SE(3) plot axes."""
        if ax is None:
            ax = plt.subplot(111, projection="3d")
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        return ax

    def add_points(self, points):
        """Add points to SE(3) object."""
        if self.point_type == "vector":
            points = SE3_VECT.matrix_from_vector(points)
        if not gs.all(SE3_MAT.belongs(points)):
            logging.warning("Some points do not belong to SE3.")
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def draw_points(self, ax, points=None, **kwargs):
        """Visualization for SE(3) points."""
        if points is None:
            points = gs.array(self.points)
        translation = points[..., :3, 3]
        frame_1 = points[:, :3, 0]
        frame_2 = points[:, :3, 1]
        frame_3 = points[:, :3, 2]
        ax.quiver(
            translation[:, 0],
            translation[:, 1],
            translation[:, 2],
            frame_1[:, 0],
            frame_1[:, 1],
            frame_1[:, 2],
            color="b",
        )
        ax.quiver(
            translation[:, 0],
            translation[:, 1],
            translation[:, 2],
            frame_2[:, 0],
            frame_2[:, 1],
            frame_2[:, 2],
            color="r",
        )
        ax.quiver(
            translation[:, 0],
            translation[:, 1],
            translation[:, 2],
            frame_3[:, 0],
            frame_3[:, 1],
            frame_3[:, 2],
            color="g",
        )
        ax.scatter(
            translation[:, 0], translation[:, 1], translation[:, 2], s=20, **kwargs
        )

    @staticmethod
    def plot_geodesic(
        initial_point, initial_tangent_vec, metric=METRIC_SE3, n_steps=40
    ):
        """Plot geodesic of SE(3).

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Point for the geodesic source.
        initial_tangent_vec : array-like, shape=[..., dim]
            Vector for the geodesic function.
        metric
            Chosen metric for geodesic
        n_steps : array-like, shape=[..., dim]
            Number of samples on the geodesic to plot.
        """
        geodesic = metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )
        t = gs.linspace(-3.0, 3.0, n_steps)
        points = geodesic(t)
        visualization.plot(points, space="SE3_GROUP")
