"""Kendall Sphere."""
import logging

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.pre_shape import KendallShapeMetric, PreShapeSpace


class KendallSphere:
    """Class used to plot points in Kendall shape space of 2D triangles.

    David G. Kendall showed that the shape space of 2D triangles is isometric
    to the 2-sphere of radius 1/2 [K1984]. This class encodes this isometric
    representation, offering a 3D visualization of Kendall shape space of order
    (3,2), and its related objects.

    Attributes
    ----------
    points : list
        List of points to plot on the Kendall sphere.
    point_type : str
        Type of the points. Can be either 'pre-shape' (for points in Kendall
        pre-shape space) or 'extrinsic' (for points given as 3x2 matrices).
        Optional, default: 'pre-shape'.
    pole : array-like, shape=[3,2]
        Equilateral triangle (north pole).
    ua : array-like, shape=[3,2]
        Tangent vector toward isocele triangle at vertex A.
    ub : array-like, shape=[3,2]
        Tangent vector toward isocele triangle at vertex B.
    na : array-like, shape=[3,2]
        Tangent vector such that (ua,na) is a positively oriented
        orthonormal basis of the horizontal space at north pole.

    References
    ----------
    .. [K1984] David G. Kendall. "Shape Manifolds, Procrustean Metrics, and
       Complex Projective Spaces." Bulletin of the London Mathematical
       Society, Volume 16, Issue 2, March 1984, Pages 81–121.
       https://doi.org/10.1112/blms/16.2.81
    """

    def __init__(self, points=None, point_type="pre-shape"):
        self.points = []
        self.point_type = point_type
        self.ax = None
        self.elev, self.azim = None, None

        self.pole = gs.array(
            [[1.0, 0.0], [-0.5, gs.sqrt(3.0) / 2.0], [-0.5, -gs.sqrt(3.0) / 2.0]]
        ) / gs.sqrt(3.0)

        self.ua = gs.array(
            [[-1.0, 0.0], [0.5, gs.sqrt(3.0) / 2.0], [0.5, -gs.sqrt(3.0) / 2.0]]
        ) / gs.sqrt(3.0)

        self.ub = gs.array(
            [[0.5, gs.sqrt(3.0) / 2.0], [0.5, -gs.sqrt(3.0) / 2], [-1.0, 0.0]]
        ) / gs.sqrt(3.0)

        self.na = self.ub - S32.ambient_metric.inner_product(self.ub, self.ua) * self.ua
        self.na = self.na / S32.ambient_metric.norm(self.na)

        if points is not None:
            self.add_points(points)

    def set_ax(self, ax=None):
        """Set axis."""
        if ax is None:
            ax = plt.subplot(111, projection="3d")

        ax_s = 0.5
        plt.setp(
            ax,
            xlim=(-ax_s, ax_s),
            ylim=(-ax_s, ax_s),
            zlim=(-ax_s, ax_s),
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )
        self.ax = ax

    def set_view(self, elev=60.0, azim=0.0):
        """Set azimuth and elevation angle."""
        if self.ax is None:
            self.set_ax()

        self.elev, self.azim = gs.pi * elev / 180, gs.pi * azim / 180
        self.ax.view_init(elev, azim)

    def convert_to_polar_coordinates(self, points):
        """Assign polar coordinates to given pre-shapes."""
        aligned_points = S32.align(points, self.pole)
        speeds = S32.ambient_metric.log(aligned_points, self.pole)

        coords_theta = gs.arctan2(
            S32.ambient_metric.inner_product(speeds, self.na),
            S32.ambient_metric.inner_product(speeds, self.ua),
        )
        coords_phi = 2.0 * S32.ambient_metric.dist(self.pole, aligned_points)

        return coords_theta, coords_phi

    def convert_to_spherical_coordinates(self, points):
        """Convert polar coordinates to spherical one."""
        coords_theta, coords_phi = self.convert_to_polar_coordinates(points)
        coords_x = 0.5 * gs.cos(coords_theta) * gs.sin(coords_phi)
        coords_y = 0.5 * gs.sin(coords_theta) * gs.sin(coords_phi)
        coords_z = 0.5 * gs.cos(coords_phi)
        spherical_coords = gs.transpose(gs.stack((coords_x, coords_y, coords_z)))
        return spherical_coords

    def add_points(self, points):
        """Add points to draw on the Kendall sphere."""
        if self.point_type == "extrinsic":
            if not gs.all(M32.belongs(points)):
                raise ValueError("Points do not belong to Matrices(3, 2).")
            points = S32.projection(points)
        elif self.point_type == "pre-shape" and not gs.all(S32.belongs(points)):
            raise ValueError("Points do not belong to the pre-shape space.")
        points = self.convert_to_spherical_coordinates(points)
        if not isinstance(points, list):
            if points.shape == (3,):
                points = [gs.array(points)]
            else:
                points = list(points)
        self.points.extend(points)

    def clear_points(self):
        """Clear the points to draw."""
        self.points = []

    def draw(self, n_theta=25, n_phi=13, scale=0.05, elev=60.0, azim=0.0):
        """Draw the sphere regularly sampled with corresponding triangles."""
        self.set_ax()
        self.set_view(elev=elev, azim=azim)
        self.ax.set_axis_off()
        plt.tight_layout()

        coords_theta = gs.linspace(0.0, 2.0 * gs.pi, n_theta)
        coords_phi = gs.linspace(0.0, gs.pi, n_phi)

        coords_x = gs.to_numpy(0.5 * gs.outer(gs.sin(coords_phi), gs.cos(coords_theta)))
        coords_y = gs.to_numpy(0.5 * gs.outer(gs.sin(coords_phi), gs.sin(coords_theta)))
        coords_z = gs.to_numpy(
            0.5 * gs.outer(gs.cos(coords_phi), gs.ones_like(coords_theta))
        )

        self.ax.plot_surface(
            coords_x,
            coords_y,
            coords_z,
            rstride=1,
            cstride=1,
            color="grey",
            linewidth=0,
            alpha=0.1,
            zorder=-1,
        )
        self.ax.plot_wireframe(
            coords_x,
            coords_y,
            coords_z,
            linewidths=0.6,
            color="grey",
            alpha=0.6,
            zorder=-1,
        )

        def lim(theta):
            return (
                gs.pi
                - self.elev
                + (2.0 * self.elev - gs.pi) / gs.pi * abs(self.azim - theta)
            )

        for theta in gs.linspace(0.0, 2.0 * gs.pi, n_theta // 2 + 1):
            for phi in gs.linspace(0.0, gs.pi, n_phi):
                if theta <= self.azim + gs.pi and phi <= lim(theta):
                    self.draw_triangle(theta, phi, scale)
                if theta > self.azim + gs.pi and phi < lim(
                    2.0 * self.azim + 2.0 * gs.pi - theta
                ):
                    self.draw_triangle(theta, phi, scale)

    def draw_triangle(self, theta, phi, scale):
        """Draw the corresponding triangle on the sphere at theta, phi."""
        u_theta = gs.cos(theta) * self.ua + gs.sin(theta) * self.na
        triangle = gs.cos(phi / 2.0) * self.pole + gs.sin(phi / 2.0) * u_theta
        triangle = scale * triangle
        triangle3d = gs.transpose(
            gs.stack((triangle[:, 0], triangle[:, 1], 0.5 * gs.ones(3)))
        )
        triangle3d = self.rotation(theta, phi) @ gs.transpose(triangle3d)

        x = list(triangle3d[0]) + [triangle3d[0, 0]]
        y = list(triangle3d[1]) + [triangle3d[1, 0]]
        z = list(triangle3d[2]) + [triangle3d[2, 0]]

        self.ax.plot3D(x, y, z, "grey", zorder=1)
        c = ["red", "green", "blue"]
        for i in range(3):
            self.ax.scatter(x[i], y[i], z[i], color=c[i], s=10, alpha=1, zorder=1)

    @staticmethod
    def rotation(theta, phi):
        """Rotation sending a triangle at pole to location theta, phi."""
        rot_th = gs.array(
            [
                [gs.cos(theta), -gs.sin(theta), 0.0],
                [gs.sin(theta), gs.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rot_phi = gs.array(
            [
                [gs.cos(phi), 0.0, gs.sin(phi)],
                [0.0, 1.0, 0.0],
                [-gs.sin(phi), 0, gs.cos(phi)],
            ]
        )
        return rot_th @ rot_phi @ gs.transpose(rot_th)

    def draw_points(self, alpha=1, zorder=0, **kwargs):
        """Draw points on the Kendall sphere."""
        points_x = [gs.to_numpy(point)[0] for point in self.points]
        points_y = [gs.to_numpy(point)[1] for point in self.points]
        points_z = [gs.to_numpy(point)[2] for point in self.points]
        self.ax.scatter(
            points_x, points_y, points_z, alpha=alpha, zorder=zorder, **kwargs
        )

    def draw_curve(self, alpha=1, zorder=0, **kwargs):
        """Draw a curve on the Kendall sphere."""
        points_x = [gs.to_numpy(point)[0] for point in self.points]
        points_y = [gs.to_numpy(point)[1] for point in self.points]
        points_z = [gs.to_numpy(point)[2] for point in self.points]
        self.ax.plot3D(
            points_x, points_y, points_z, alpha=alpha, zorder=zorder, **kwargs
        )

    def draw_vector(self, tangent_vec, base_point, **kwargs):
        """Draw one vector in the tangent space to sphere at a base point."""
        norm = METRIC_S32.norm(tangent_vec, base_point)
        exp = METRIC_S32.exp(tangent_vec, base_point)
        bp = self.convert_to_spherical_coordinates(base_point)
        exp = self.convert_to_spherical_coordinates(exp)
        tv = exp - gs.dot(exp, 2.0 * bp) * 2.0 * bp
        tv = tv / gs.linalg.norm(tv) * norm
        self.ax.quiver(bp[0], bp[1], bp[2], tv[0], tv[1], tv[2], **kwargs)
