"""Visualization for Geometric Statistics."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import geomstats.backend as gs


class Ellipses:
    """Class used to plot points on the manifold SPD(2).

    Elements S of the manifold of 2D Symmetric Positive Definite matrices
    can be conveniently represented by ellipses.

    We write :math: `S = O D O^T` with :math: `O` an orthogonal matrix (rotation)
    and :math: `D` a diagonal matrix. The positive eigenvalues, i.e. the elements of
    :math: `D`, are the inverse of the length of the major and minor axes of
    the ellipse. The rotation matrix :math: `O` determines the orientation of the
    2D ellipse in the 2D plane.

    Parameters
    ----------
    n_sampling_points : int
        Number of points to sample on the discretized ellipses.
    """

    def __init__(self, n_sampling_points=100):
        self.n_sampling_points = n_sampling_points

    @staticmethod
    def set_ax(ax=None):
        """Set the axis for the Figure.

        Parameters
        ----------
        ax : Axis
            Axis of the figure.

        Returns
        -------
        ax : Axis
            Axis of the figure.
        """
        if ax is None:
            ax = plt.subplot()
        plt.setp(ax, xlabel="X", ylabel="Y")
        return ax

    def draw_points(self, points=None, ax=None, **plot_kwargs):
        """Draw the ellipses.

        Parameters
        ----------
        ax : Axis
            Axis of the figure.
        points : array-like, shape=[..., 2, 2]
            Points on the SPD manifold of 2D symmetric
            positive definite matrices.
            Optional, default: None.
        plot_kwargs : dict
            Dictionnary of arguments related to plotting.
        """
        if ax is None:
            ax = self.set_ax()
        if points.ndim == 2:
            points = [points]
        for point in points:
            x_coords, y_coords = self.compute_coordinates(point)
            ax.plot(x_coords, y_coords, **plot_kwargs)

    def compute_coordinates(self, point):
        """Compute the ellipse coordinates of a 2D SPD matrix.

        Parameters
        ----------
        point : array-like, shape=[2, 2]
            SPD matrix.

        Returns
        -------
        x_coords : array-like, shape=[n_sampling_points,]
            x_coords coordinates of the sampling points on the discretized ellipse.
        Y: array-like, shape = [n_sampling_points,]
            y coordinates of the sampling points on the discretized ellipse.
        """
        eigvalues, eigvectors = gs.linalg.eigh(point)
        eigvalues = gs.where(eigvalues < gs.atol, gs.atol, eigvalues)

        [eigvalue1, eigvalue2] = eigvalues

        rot_sin = eigvectors[1, 0]
        rot_cos = eigvectors[0, 0]
        thetas = gs.linspace(0.0, 2 * gs.pi, self.n_sampling_points + 1)

        x_coords = eigvalue1 * gs.cos(thetas) * rot_cos
        x_coords -= rot_sin * eigvalue2 * gs.sin(thetas)
        y_coords = eigvalue1 * gs.cos(thetas) * rot_sin
        y_coords += rot_cos * eigvalue2 * gs.sin(thetas)
        return x_coords, y_coords
