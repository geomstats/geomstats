"""Visualization for Geometric Statistics.

consolidated version

Lead authors: past
"""


import logging

import geomstats.backend as gs
import geomstats.visualization as visualization
import matplotlib.pyplot as plt
from geomstats.geometry.special_euclidean import SpecialEuclidean
from mpl_toolkits.mplot3d import Axes3D  # NOQA

SE2_GROUP = SpecialEuclidean(n=2, point_type="matrix")
SE2_VECT = SpecialEuclidean(n=2, point_type="vector")
METRIC2 = SE2_VECT.left_canonical_metric


class SpecialEuclidean2:
    """Class used to plot points in the 2d special euclidean group.

    The points are stored in the vector form by default.
    If the point is generated in the matrix form, it will be
    transformed into the vector form and stored.
    The plotting is based on the matrix form, so the points are
    transformed into the matrix form before being plotted.
    """

    def __init__(self, points=None, point_type="vector"):
        self.n = 2
        self.points = []
        self.point_type = point_type
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None, **kwargs):
        """Set the axes attributes before plotting.

        Keywords:
            x_lim, x_label,
            y_lim, y_label,
            fontweight, fontsize
        """
        if ax is None:
            ax = plt.subplot()

        fontdict = {}
        if "fontweight" in kwargs.keys():
            fontdict["fontweight"] = kwargs["fontweight"]
        if "fontsize" in kwargs.keys():
            fontdict["fontsize"] = kwargs["fontsize"]

        if "x_lim" in kwargs.keys():
            ax.set_xlim(kwargs["x_lim"])
        if "y_lim" in kwargs.keys():
            ax.set_ylim(kwargs["y_lim"])
        if "x_label" in kwargs.keys():
            ax.set_xlabel(kwargs["x_label"], **fontdict)
        if "y_label" in kwargs.keys():
            ax.set_ylabel(kwargs["y_label"], **fontdict)

        return ax

    def add_points(self, points):
        """Add SE2 points into the class.

        If the class is used to process the matrix form, points
        will be converted into the vector form first.

        Parameter
        ----------
            points: array-like, shape=[..., 3,3] (matrix)
                                    or [..., 3] (vector)
        """
        if self.point_type == "matrix":
            points = SE2_VECT.vector_from_matrix(points)
        if not gs.all(SE2_VECT.belongs(points)):
            logging.warning("Some points do not belong to SE2.")
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def draw_points(self, ax, points=None, point_type="vector", **kwargs):
        """Plot the SE2 points in the matrix form.

        Parameters
        ----------
        points : array-like, shape=[..., 3, 3] (matrix)
                                or [..., 3] (vector)
            Points to be plotted.
        point_type: specify the representation of the points if they are passed
            into the method.
            If the points are in the matrix form, specify here so that they won't
            be converted into the matrix again.
        """
        if points is None:
            points = gs.array(self.points)
            points = SE2_VECT.matrix_from_vector(points)
        elif point_type == "vector":
            points = SE2_VECT.matrix_from_vector(points)

        translation = points[..., : self.n, self.n]
        frames = points[..., : self.n, : self.n]
        color_list = ["b", "r"]

        for d in range(self.n):
            ax.quiver(
                translation[..., 0],
                translation[..., 1],
                frames[..., 0, d],
                frames[..., 1, d],
                width=0.005,
                color=color_list[d],
            )
        ax.scatter(translation[..., 0], translation[..., 1], s=16, **kwargs)

    def plot_geodesic(self, point, vector, n_steps, ax=None, **kwargs):
        """Plot a geodesic of SE2.

        SE2 is equipped with its left-invariant canonical metric.

        Parameters
        ----------
        point : array-like, shape=[..., dim] (vector)
            Point for the geodesic function.
        vector : array-like, shape=[..., dim]
            Vector for the geodesic function.
        n_steps : integer
            Number of samples on the geodesic to plot.
        """
        # passes in a point and vector to the geodesic function
        # of the left canonical matrix type
        initial_point = point
        initial_tangent_vec = gs.array(vector)
        geodesic = METRIC2.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )

        # defines the sampling of points on the geodesic
        t = gs.linspace(-3.0, 3.0, n_steps)

        points = geodesic(t)

        # creates figure
        ax = self.set_ax(ax=ax, **kwargs)

        self.draw_points(ax, points, point_type="vector", **kwargs)


SE3_GROUP = SpecialEuclidean(n=3, point_type="matrix")
SE3_VECT = SpecialEuclidean(n=3, point_type="vector")
METRIC3 = SE3_VECT.left_canonical_metric


class SpecialEuclidean3:
    """Class used to plot points in the 3d special euclidean group.

    The points are stored in the vector form by default.
    If the point is generated in the matrix form, it will be
    transformed into the vector form and stored.
    The plotting is based on the matrix form, so the points are
    transformed into the matrix form before being plotted.
    """

    def __init__(self, points=None, point_type="vector"):
        self.n = 3
        self.points = []
        self.point_type = point_type
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None, **kwargs):
        """Set the axes attributes before plotting.

        Keywords:
            x_lim, x_label,
            y_lim, y_label,
            z_lim, z_label,
            fontweight, fontsize
        """
        if ax is None:
            ax = plt.subplot(111, projection="3d")

        fontdict = {}
        if "fontweight" in kwargs.keys():
            fontdict["fontweight"] = kwargs["fontweight"]
        if "fontsize" in kwargs.keys():
            fontdict["fontsize"] = kwargs["fontsize"]

        if "x_lim" in kwargs.keys():
            ax.set_xlim(kwargs["x_lim"])
        if "y_lim" in kwargs.keys():
            ax.set_ylim(kwargs["y_lim"])
        if "z_lim" in kwargs.keys():
            ax.set_zlim(kwargs["z_lim"])

        if "x_label" in kwargs.keys():
            ax.set_xlabel(kwargs["x_label"], **fontdict)
        if "y_label" in kwargs.keys():
            ax.set_ylabel(kwargs["y_label"], **fontdict)
        if "z_label" in kwargs.keys():
            ax.set_zlabel(kwargs["z_label"], **fontdict)

        return ax

    def add_points(self, points):
        """Add SE3 points into the class.

        If the class is used to process the matrix form, points
        will be converted into the vector form first.

        Parameter
        ----------
            points: array-like, shape=[..., 4,4] (matrix)
                                    or [..., 6] (vector)
        """
        if self.point_type == "matrix":
            points = SE3_VECT.vector_from_matrix(points)
        if not gs.all(SE3_VECT.belongs(points)):
            logging.warning("Some points do not belong to SE3.")
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def draw_points(self, ax, points=None, point_type="vector", **kwargs):
        """Plot the SE3 points in the vector form.

        Parameters
        ----------
        points : array-like, shape=[..., 4, 4] (matrix)
                                or [..., 6] (vector)
            Points to be plotted.
        point_type: specify the representation of the points if they are passed
            into the method.
            If the points are in the matrix form, specify here so that they are
            transformed into the vector form.
        """
        if points is None:
            points = gs.array(self.points)
        elif point_type == "matrix":
            points = SE3_VECT.vector_from_matrix(points)

        visualization.plot(points, ax=ax, space="SE3_GROUP", **kwargs)

    def plot_geodesic(self, point, vector, n_steps, ax=None, **kwargs):
        """Plot a geodesic of SE3.

        SE2 is equipped with its left-invariant canonical metric.

        Parameters
        ----------
        point : array-like, shape=[..., dim] (vector)
            Point for the geodesic function.
        vector : array-like, shape=[..., dim]
            Vector for the geodesic function.
        n_steps : integer
            Number of samples on the geodesic to plot.
        """
        initial_point = point
        initial_tangent_vec = gs.array(vector)
        geodesic = METRIC3.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )

        t = gs.linspace(-3.0, 3.0, n_steps)
        points = geodesic(t)

        self.draw_points(ax=ax, points=points, **kwargs)
