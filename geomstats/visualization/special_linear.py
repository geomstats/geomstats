"""Visualization for Geometric Statistics."""

from functools import partial

import geomstats.backend as gs
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.special_linear import SpecialLinear
from mpl_toolkits.mplot3d import Axes3D  # NOQA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SL2 = SpecialLinear(n=2)
SL3 = SpecialLinear(n=3)

AX_SCALE = 1.2


class Squares:
    """Class used to draw the unit square and transformations using SL2(R)."""

    def __init__(self, num_points=10, points=None):
        """Squares() constructor.

        Construct an instance of the Squares() class.

        Parameters
        ----------
        num_points : int
            The number of points to use for the representation of the unit
            square along one axis
        points : array-like, shape=[..., 2, 2]
            Points on the SL2(R) manifold of 2D matrices with a determinant of
            1.
        """
        self.num_points = num_points
        x = gs.linspace(-0.5, 0.5, num_points)
        y = gs.linspace(-0.5, 0.5, num_points)
        xcoord, ycoord = gs.meshgrid(x, y)
        self.unit_X = gs.reshape(xcoord, (xcoord.size, 1))
        self.unit_Y = gs.reshape(ycoord, (ycoord.size, 1))
        self.points = []
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None):
        """Set the axes for plotting.

        Parameters
        ----------
        ax : matplotlib.axes object
            Axis of the figure.

        Returns
        -------
        ax : matplotlib.axes object
            Axis of the figure modified with x and y limits.
        """
        if ax is None:
            fig = plt.gcf()
            fig.clear()
            ax = fig.add_subplot(111)
        ax_s = AX_SCALE
        plt.setp(ax, xlim=(-ax_s, ax_s), ylim=(-ax_s, ax_s), xlabel="X", ylabel="Y")
        return ax

    def transformation_2D(self, matrix):
        """Compute the 2D transformation of the unit square using matrix.

        Parameters
        ----------
        matrix : array-like, shape = [2, 2]
            2D matrix transformation that is a member of SL2(R).

        Returns
        -------
        X_trans : array-like, shape = [..., 1]
            The transformed x-coordinates of the unit square.
        Y_trans : array-like, shape = [..., 1]
            The transformed y-coordinates of the unit square.
        """
        X_trans = gs.empty(self.num_points**2)
        Y_trans = gs.empty(self.num_points**2)
        for i in range(self.num_points**2):
            X_trans[i] = Matrices.mul(
                matrix, gs.array([self.unit_X[i], self.unit_Y[i]])
            )[0]
            Y_trans[i] = Matrices.mul(
                matrix, gs.array([self.unit_X[i], self.unit_Y[i]])
            )[1]
        return X_trans, Y_trans

    def add_points(self, points):
        """Add points to Squares()."""
        if not gs.all(SL2.belongs(points)):
            raise ValueError("Points do  not belong to SL2.")
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def return_frame(self, time, ax=None):
        """Create a frame of the animation.

        Parameters
        ----------
        time : int
            Parameter along the transformation trajectory saved in the Square
            object.
        ax : matplotlib.axes object
            Axis of the figure.
        """
        ax.clear()
        self.set_ax(ax=ax)
        ax.scatter(
            self.transformation_2D(self.points[time])[0],
            self.transformation_2D(self.points[time])[1],
            c=gs.linspace(0, 1, 100),
            cmap=plt.cm.jet,
        )

    def animate(self, points, ax=None, **point_draw_kwargs):
        """Animation of the unit square.

        Create an animation of the transformations of the unit square using
        the SL2(R) transformations stored in self.points.

        Parameters
        ----------
        points : array-like, shape=[..., 2, 2]
            Points on the SL2(R) manifold of 2D matrices with a determinant of
            1.
        ax : matplotlib.axes object
            Axis of the figure.
        """
        ax = self.set_ax()
        self.add_points(points)
        frame = partial(self.return_frame, ax=ax)
        movie = animation.FuncAnimation(
            plt.gcf(), frame, interval=100, frames=len(self.points), blit=False
        )
        plt.show()
        return movie


class Cubes:
    """Class used to draw the unit cube and transformations using SL3(R)."""

    def __init__(self, points=None):
        """Cubes() constructors.

        Construct an instance of the Cubes() class.

        Parameters
        ----------
        points : array-like, shape=[..., 3, 3]
            Points on the SL3(R) manifold of 3D matrices with a determinant of
            1.
        """
        self.unit_vertices = gs.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ]
        )
        self.points = []
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None):
        """Set the axes for plotting.

        Parameters
        ----------
        ax : matplotlib.axes object
            Axis of the figure.

        Returns
        -------
        ax : matplotlib.axes object
            Axis of the figure modified with x, y, and z limits.
        """
        if ax is None:
            fig = plt.gcf()
            fig.clear()
            ax = fig.add_subplot(111, projection="3d")
        ax_s = AX_SCALE
        plt.setp(
            ax,
            xlim=(-ax_s, ax_s),
            ylim=(-ax_s, ax_s),
            zlim=(-ax_s, ax_s),
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )
        return ax

    def transformation_3D(self, matrix):
        """Compute the 3D transformation of the unit cube using matrix.

        Parameters
        ----------
        matrix : array-like, shape = [3, 3]
            3D matrix transformation that is a member of SL3(R).

        Returns
        -------
        X_trans : array-like, shape = [..., 1]
            The transformed x-coordinates of the vertices of the unit cube.
        Y_trans : array-like, shape = [..., 1]
            The transformed y-coordinates of the vertices of the unit cube.
        Z_trans : array-like, shape = [..., 1]
            The transformed z-coordinates of the vertices of the unit cube.
        """
        X_trans = gs.empty(self.unit_vertices.shape[0])
        Y_trans = gs.empty(self.unit_vertices.shape[0])
        Z_trans = gs.empty(self.unit_vertices.shape[0])
        for i in range(self.unit_vertices.shape[0]):
            vertices_trans = Matrices.mul(matrix, gs.transpose([self.unit_vertices[i]]))
            X_trans[i] = vertices_trans[0]
            Y_trans[i] = vertices_trans[1]
            Z_trans[i] = vertices_trans[2]
        return X_trans, Y_trans, Z_trans

    def add_points(self, points):
        """Add points to Cubes()."""
        if not gs.all(SL3.belongs(points)):
            raise ValueError("Points do  not belong to SL3.")
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def return_frame(self, time, ax=None):
        """Create a frame of the animation.

        Parameters
        ----------
        time : int
            Parameter along the transformation trajectory saved in the Cubes
            object.
        ax : matplotlib.axes object
            Axis of the figure.
        """
        ax.clear()
        ax = self.set_ax(ax=ax)
        pts = self.transformation_3D(self.points[time])
        X_trans = pts[0]
        Y_trans = pts[1]
        Z_trans = pts[2]
        ax.scatter3D(X_trans, Y_trans, Z_trans)
        pts_T = gs.transpose(pts)
        edges = [
            [pts_T[0], pts_T[1], pts_T[2], pts_T[3]],
            [pts_T[4], pts_T[5], pts_T[6], pts_T[7]],
            [pts_T[0], pts_T[1], pts_T[5], pts_T[4]],
            [pts_T[2], pts_T[3], pts_T[7], pts_T[6]],
            [pts_T[1], pts_T[2], pts_T[6], pts_T[5]],
            [pts_T[4], pts_T[7], pts_T[3], pts_T[0]],
        ]
        ax.add_collection3d(
            Poly3DCollection(
                edges, facecolors="cyan", linewidths=1.0, edgecolors="b", alpha=0.2
            )
        )

    def animate(self, points, ax=None, **point_draw_kwargs):
        """Animation of the unit cube.

        Create an animation of the transformations of the unit cube using the
        SL3(R) transformations stored in self.points.

        Parameters
        ----------
        points : array-like, shape=[..., 3, 3]
            Points on the SL3(R) manifold of 3D matrices with a determinant of
            1.
        ax : matplotlib.axes object
            Axis of the figure.
        """
        ax = self.set_ax(ax=ax)
        self.add_points(points)
        frame = partial(self.return_frame, ax=ax)
        movie = animation.FuncAnimation(
            plt.gcf(), frame, interval=100, frames=len(self.points), blit=False
        )
        plt.show()
        return movie
