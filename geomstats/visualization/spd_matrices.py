"""Functions for SPD Visualization."""

import colorsys
import math

import matplotlib.pyplot as plt
import numpy as np
from geomstats.geometry.spd_matrices import SPDAffineMetric, SPDMatrices
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # NOQA
import geomstats.backend as gs

class Ellipses:
    """Class used to plot points on the manifold SPD(2).
    Elements S of the manifold of 2D Symmetric Positive Definite matrices
    can be conveniently represented by ellipses.
    We write :math:`S = O D O^T` with :math:`O` an orthogonal matrix (rotation)
    and :math:`D` a diagonal matrix. The positive eigenvalues, i.e. the elements of
    :math:`D`, are the inverse of the length of the major and minor axes of
    the ellipse. The rotation matrix :math:`O` determines the orientation of the
    2D ellipse in the 2D plane.
    Parameters
    ----------
    k_sampling_points : int
        Number of points to sample on the discretized ellipses.
    """

    def __init__(self, k_sampling_points=100):
        self.k_sampling_points = k_sampling_points

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
        x_coords : array-like, shape=[k_sampling_points,]
            x_coords coordinates of the sampling points on the discretized ellipse.
        Y: array-like, shape = [k_sampling_points,]
            y coordinates of the sampling points on the discretized ellipse.
        """
        eigvalues, eigvectors = gs.linalg.eigh(point)
        eigvalues = gs.where(eigvalues < gs.atol, gs.atol, eigvalues)

        [eigvalue1, eigvalue2] = eigvalues

        rot_sin = eigvectors[1, 0]
        rot_cos = eigvectors[0, 0]
        thetas = gs.linspace(0.0, 2 * gs.pi, self.k_sampling_points + 1)

        x_coords = eigvalue1 * gs.cos(thetas) * rot_cos
        x_coords -= rot_sin * eigvalue2 * gs.sin(thetas)
        y_coords = eigvalue1 * gs.cos(thetas) * rot_sin
        y_coords += rot_cos * eigvalue2 * gs.sin(thetas)
        return x_coords, y_coords

class SPDMatricesViz:
    """Class for the visualization of the manifold for SPD matrices.

    This class provides all the essential methods for the
    visualization of the manifold of the Symmetric Positive
    Definite matrices.

    Parameters
    ----------
    max_z: int
    The scaling factor of the manifold

    Attributes
    ----------
    curr_z: int
    The scaling factor of the manifold

    spd_point_viz:
    Class used to plot points on the manifold SPD(2).
    Elements S of the manifold of 2D Symmetric Positive
    Definite matrices can be conveniently represented by ellipses.

    spd_manifold:
    Class for the manifold of symmetric positive definite (SPD) matrices.
    Takes as input n (int) Integer representing the shape of the matrices: nxn
    By default n is set to 2 to in order for the visualization to be feasible.

    metric:
    Compute the affine-invariant exponential map.Compute the Riemannian
    exponential at point base_point of tangent vector tangent_vec wrt
    the metric defined in inner_product.
    This gives a symmetric positive definite matrix.

    References
    ----------
    [1] Miolane, Nina, et al. "Geomstats: a Python package for
    Riemannian geometry in machine learning."
    Journal of Machine Learning Research 21.223 (2020): 1-9.
    """

    def __init__(self, max_z=1):
        """Initialize class parameters of the manifold."""
        self.max_z = float(max_z)
        self.curr_z = self.max_z
        self.ax = None
        self.spd_point_viz = Ellipses()
        self.spd_manifold = SPDMatrices(2)
        self.metric = SPDAffineMetric(2)

    def cuboid_data(self, pos, size=(1, 1, 1)):
        """Generate cuboid data for plotting.
        
        Parameters
        ----------
        pos : tuple
            Position coordinates in Eucldiean space
        size : tuple
            Size of cube in each dimension

        Returns
        -------
        cuboid_array : Array of cuboid data
        """
        cuboid_array = [
            [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
            [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
        ]
        cuboid_array = np.array(cuboid_array).astype(float)
        cuboid_array -= 0.5
        for i in range(3):
            cuboid_array[:, :, i] *= size[i]
        cuboid_array += np.array(pos)
        return cuboid_array

    def plot_cube_at(self, positions, sizes=None, colors=None, **kwargs):
        """Plot cube relative to specific coordinates.

        Parameters
        ----------
        positions : array-like, size [1,3]
            Cordinates of the specific point
        sizes : list of tuples
            Size of the cube-shaped tangent space
        colors: string, optional (default=None)
            Specifies the color of the cube

        Returns
        -------
            Figure plot
        """
        if not isinstance(colors, (list, np.ndarray)):
            colors = ["C0"] * len(positions)
        if not isinstance(sizes, (list, np.ndarray)):
            sizes = [(1, 1, 1)] * len(positions)
        generated_cuboid = []
        for p, s, c in zip(positions, sizes, colors):
            generated_cuboid.append(self.cuboid_data(p, size=s))
        return Poly3DCollection(
            np.concatenate(generated_cuboid),
            facecolors=np.repeat(colors, 6, axis=0),
            **kwargs
        )

    def plot(self, n_angles=80, n_radii=40, curr_z=None, hsv=False):
        """Plot the 3D cone.

        Parameters
        ----------
        n_angles : int
            Number of angles in polar coordinates
        n_radii : int
            Number of radii in polar coordinates
        curr_z: int, optional (default=None)
            Scaling factor
        hsv: bool, optional (default=False)
            Adds smooth gradient representation to the cone when set to True

        Returns
        -------
            Figure plot
        """
        if curr_z is None:
            self.curr_z = self.max_z
        else:
            self.curr_z = curr_z

        radii = np.linspace(0.0, self.curr_z, n_radii)
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

        # Convert polar (radii, angles) coords to cartesian (x, y) coords
        # (0, 0) is added here.
        # There are no duplicate points in the (x, y) plane
        x = np.append(0, (radii * np.cos(angles)).flatten())
        y = np.append(0, (radii * np.sin(angles)).flatten())

        # # Pringle surface
        z = np.full_like(x, self.curr_z)

        # print(x.shape, y.shape, angles.shape, radii.shape, z.shape)
        # # NOTE: This assumes that there is a nice projection of
        # the surface into the x/y-plane!
        tri = Triangulation(x, y)
        triangle_vertices = np.array(
            [
                np.array(
                    [
                        [x[T[0]], y[T[0]], z[T[0]]],
                        [x[T[1]], y[T[1]], z[T[1]]],
                        [x[T[2]], y[T[2]], z[T[2]]],
                    ]
                )
                for T in tri.triangles
            ]
        )
        x2 = np.append(0, (radii * np.cos(angles)).flatten())
        y2 = np.append(0, (radii * np.sin(angles)).flatten())

        # Pringle surface
        z2 = np.sqrt(x**2 + y**2)

        # NOTE: This assumes that there is a nice projection
        # of the surface into the x/y-plane!
        tri2 = Triangulation(x2, y2)

        triangle_vertices2 = np.array(
            [
                np.array(
                    [
                        [x2[T[0]], y2[T[0]], z2[T[0]]],
                        [x2[T[1]], y2[T[1]], z2[T[1]]],
                        [x2[T[2]], y2[T[2]], z2[T[2]]],
                    ]
                )
                for T in tri2.triangles
            ]
        )

        triangle_vertices = np.concatenate(
            [triangle_vertices, triangle_vertices2])
        midpoints = np.average(triangle_vertices, axis=1)

        if hsv:
            facecolors = [
                self.find_color_for_point(pt) for pt in midpoints
            ]  # smooth gradient
        else:
            facecolors = "0.9"  # grey

        coll = Poly3DCollection(
            triangle_vertices,
            facecolors=facecolors,
            edgecolors=None,
            alpha=0.5,
            zorder=-10,
        )
        self.artist = coll
        self.fig = plt.figure()

        self.ax = self.fig.add_subplot(projection="3d")
        self.ax.add_collection(coll)

        self.ax.set_xlim(-self.max_z * 1.25, self.max_z * 1.25)
        self.ax.set_ylim(-self.max_z * 1.25, self.max_z * 1.25)
        self.ax.set_zlim(-self.max_z * 0.25, self.max_z * 1.25)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.elev = 26

    @staticmethod
    def elms_to_xyz(point):
        """Convert elm geometry coordinates to Cartesian coordinates.

        Parameters
        ----------
        point : tuple-like of size = 3

        Returns
        -------
        point : tuple-like of size = 3
        """
        elm0, elm1, elm2 = point
        z = (elm0 + elm2) / 2
        y = elm1
        x = elm0 - z
        return (x, y, z)

    @staticmethod
    def xyz_to_elms(point):
        """Convert Cartesian coordinates to elm geometry coordinates.

        Parameters
        ----------
        point : tuple-like of size = 3

        Returns
        -------
        point : tuple-like of size = 3
        """
        x, y, z = point
        elm0 = z + x
        elm1 = y
        elm2 = z - x
        return (elm0, elm1, elm2)

    @staticmethod
    def xyz_to_spd(point):
        """Convert Cartesian coordinates to coordinates on the manifold.

        Parameters
        ----------
        point : tuple-like of size = 3

        Returns
        -------
        matrix: array-like, shape [2,2]
        """
        x, y, z = point
        return np.array([[z + x, y], [y, z - x]])

    @staticmethod
    def spd_to_xyz(matrix):
        """Convert coordinates on manifold to Cartesian coordinates.

        Parameters
        ----------
        matrix: array-like, shape [2,2]

        Returns
        -------
        point : tuple-like of size = 3
        """
        z = (matrix[0, 0] + matrix[1, 1]) / 2.0
        x = matrix[0, 0] - z
        y = matrix[0, 1]

        return (x, y, z)

    def find_color_for_point(self, point):
        """Convert the color from HSV coordinates to RGB coordinates.

        Parameters
        ----------
        point : tuple-like of size = 3

        Returns
        -------
        color: tuple-like of size = 3
        """
        x, y, z = point

        # convert radians to degrees
        angle = np.arctan2(x, y) * 180 / np.pi

        # normalize degrees to [0, 360]
        if angle < 0:
            angle = angle + 360

        hue = angle / 360
        saturation = math.sqrt(x**2 + y**2) / self.max_z
        value = z / self.max_z

        color = colorsys.hsv_to_rgb(hue, saturation, value)

        return color

    def plot_grid(self):
        """Plot the geodesic grid.

        Returns
        -------
            Figure plot
        """
        self.plot_geodesic(
            startpt_xyz=(0, 0, 0.5),
            endpt_xyz=(0, 0, 0.6),
            n_geodesic_samples=30
        )
        self.plot_geodesic(
            startpt_xyz=(0, 0, 0.5),
            endpt_xyz=(0, 0.1, 0.5),
            n_geodesic_samples=30
        )
        self.plot_geodesic(
            startpt_xyz=(0, 0, 0.5),
            endpt_xyz=(0.1, 0, 0.5),
            n_geodesic_samples=30
        )

    def plot_rendering_top(self, n_radii, n_angles):
        """Plot ellipses on the top of the cone manifold.

        Parameters
        ----------
        n_angles : int
            Number of angles in polar coordinates
        n_radii : int
            Number of radii in polar coordinates

        Returns
        -------
        Figure plot
        """

        z_plane = self.curr_z
        radii = np.linspace(z_plane, 0, n_radii, endpoint=False)
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

        x = np.append(0, (radii * np.cos(angles)).flatten())
        y = np.append(0, (radii * np.sin(angles)).flatten())

        for x_tmp, y_tmp in zip(x, y):
            sampled_xyz = (x_tmp, y_tmp, z_plane)
            sampled_spd = self.xyz_to_spd(sampled_xyz)
            ellipse_x, ellipse_y = self.spd_point_viz.compute_coordinates(
                sampled_spd)
            self.ax.plot(
                ellipse_x / (n_radii * n_angles * 0.25) + sampled_xyz[0],
                ellipse_y / (n_radii * n_angles * 0.25) + sampled_xyz[1],
                sampled_xyz[2],
                alpha=0.8,
                zorder=10,
                color=self.find_color_for_point(sampled_xyz),
            )

    def plot_rendering(self, n_radii=5, n_angles=16):
        """Draw the manifold with regularly sampled data.

        Parameters
        ----------
        n_radii : int
            Number of radii in polar coordinates
        n_angles: int
            Number of angles in polar coordinates
        Returns
        -------
         Figure plot
        """
        self.ax.elev = 90
        self.plot_rendering_top(n_radii, n_angles)

    def plot_tangent_space(self, point):
        """Plot the tangent space of the SPD manifold.

        Needs a given set of coordinates on the manifold

        Parameters
        ----------
        point : tuple-like, size 3
            Coordinates of the point based on which
            the tangent space will be plotted

        Returns
        -------
        Figure plot
        """
        x, y, z = point

        positions = np.array([[x, y, z]])
        pc = self.plot_cube_at(
            positions,
            sizes=[(0.1, 0.1, 0.1 * 0.5)] * len(positions),
            edgecolor="k",
            alpha=0.8,
            zorder=10,
        )
        self.ax.add_collection3d(pc)

    def scatter(self, n_samples=100):
        """Plot point cloud according to the manifold.

        Parameters
        ----------
        n_samples : int
            Number of samples to be scattered

        Returns
        -------
        Figure plot
        """
        list_of_transf_samples = []
        samples = self.spd_manifold.random_point(n_samples=n_samples)
        for i in samples:
            transf_sample = list(self.spd_to_xyz(i))
            list_of_transf_samples.append(transf_sample)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        list_of_transf_samples = np.array(list_of_transf_samples)

        xs = list_of_transf_samples[:, 0]
        ys = list_of_transf_samples[:, 1]
        zs = list_of_transf_samples[:, 2]
        ax.scatter(xs, ys, zs, marker="o")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return fig

    def plot_exp(self,
                 startpt_xyz=(0, 0, 1),
                 tangentVectorXYZ=(0.5, 0.5, -0.25)):
        """Plot exponential map of the manifold.

        Parameters
        ----------
        startpt_xyz : tuple-like, size 3
            Coordinates of starting point
        tangentVectorXYZ : tuple-like, size 3
            Vector that define the arrow direction and location

        Returns
        -------
        Figure plot
        """
        tangent_matrix = self.xyz_to_spd(
            tangentVectorXYZ)

        print("Tangent Matrix")
        print(tangent_matrix)

        self.ax.scatter3D(
            startpt_xyz[0], startpt_xyz[1], startpt_xyz[2], label="Start Point"
        )
        self.ax.quiver(
            startpt_xyz[0],
            startpt_xyz[1],
            startpt_xyz[2],
            tangentVectorXYZ[0],
            tangentVectorXYZ[1],
            tangentVectorXYZ[2],
            label="Tangent Vector",
        )

        result_matrix = self.metric.exp(
            tangent_matrix,
            base_point=self.xyz_to_spd(startpt_xyz),
        )

        print("Result")
        print(result_matrix)
        result_xyz = self.spd_to_xyz(result_matrix)
        self.ax.scatter3D(
            result_xyz[0], result_xyz[1], result_xyz[2], label="Result: Point"
        )
        self.ax.legend()

    def plot_log(self, startpt_xyz=(0, 0, 1), endpt_xyz=(0.25, 0.25, 0.5)):
        """Plot logarithmic map of the manifold.

        Parameters
        ----------
        startpt_xyz : tuple-like, size 3
            Coordinates of starting point
        endpt_xyz : tuple-like, size 3
            Coordinates of ending point

        Returns
        -------
        Figure plot
        """
        tangent_matrix = self.metric.log(
            self.xyz_to_spd(endpt_xyz),
            base_point=self.xyz_to_spd(startpt_xyz),
        )

        tangent_vector = self.spd_to_xyz(tangent_matrix)

        self.ax.scatter3D(
            startpt_xyz[0], startpt_xyz[1], startpt_xyz[2], label="Start Point"
        )
        self.ax.scatter3D(
            endpt_xyz[0], endpt_xyz[1], endpt_xyz[2], label="End Point")
        self.ax.quiver(
            startpt_xyz[0],
            startpt_xyz[1],
            startpt_xyz[2],
            tangent_vector[0],
            tangent_vector[1],
            tangent_vector[2],
            label="Result: Tangent Vector",
        )
        self.ax.legend()

    def plot_geodesic(
        self,
        startpt_xyz=(0, 0, 1),
        endpt_xyz=(0.25, 0.25, 0.5),
        n_geodesic_samples=30,
    ):
        """Plot the discretised geodesic of the manifold.

        Takes either point and tangent vec as parameters,
        or initial point and end point as parameters.

        Parameters
        ----------
        startpt_xyz : tuple-like, size 3
            Initial point of the geodesic
        endpt_xyz : tuple-like, size 3
            End point of the geodesic
        n_geodesic_sample: int
            Number of samples for discretization

        Returns
        -------
        Figure plot
        """
        tangent_matrix = self.metric.log(
            self.xyz_to_spd(endpt_xyz),
            base_point=self.xyz_to_spd(startpt_xyz),
        )

        tangent_vector = self.spd_to_xyz(tangent_matrix)

        self.ax.scatter3D(
            startpt_xyz[0],
            startpt_xyz[1],
            startpt_xyz[2],
            label="Start Point"
        )
        self.ax.scatter3D(
            endpt_xyz[0],
            endpt_xyz[1],
            endpt_xyz[2],
            label="End Point")
        self.ax.quiver(
            startpt_xyz[0],
            startpt_xyz[1],
            startpt_xyz[2],
            tangent_vector[0],
            tangent_vector[1],
            tangent_vector[2],
            label="Tangent Vector",
        )

        result = self.metric.geodesic(
            initial_tangent_vec=tangent_matrix,
            initial_point=self.xyz_to_spd(startpt_xyz),
        )

        points_on_geodesic_spd = result(
            np.linspace(0.0, 1.0, n_geodesic_samples))

        geodesicXYZ = np.zeros((n_geodesic_samples, 3))
        pointColors = []
        for index, matrix in enumerate(points_on_geodesic_spd):
            geodesicXYZ[index, :] = self.spd_to_xyz(matrix)
            pointColors.append(
                self.find_color_for_point(geodesicXYZ[index, :]))

        self.ax.scatter3D(
            geodesicXYZ[1:-1, 0],
            geodesicXYZ[1:-1, 1],
            geodesicXYZ[1:-1, 2],
            alpha=1,
            edgecolors="black",
            color=pointColors[1:-1],
            label="Discrete Geodesic",
            zorder=100,
        )
        self.ax.legend()
