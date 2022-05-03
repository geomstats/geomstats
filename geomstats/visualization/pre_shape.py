"""Visualization for Geometric Statistics."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.pre_shape import KendallShapeMetric, PreShapeSpace

M32 = Matrices(m=3, n=2)
S32 = PreShapeSpace(k_landmarks=3, m_ambient=2)
METRIC_S32 = KendallShapeMetric(k_landmarks=3, m_ambient=2)
M33 = Matrices(m=3, n=3)
S33 = PreShapeSpace(k_landmarks=3, m_ambient=3)
METRIC_S33 = KendallShapeMetric(k_landmarks=3, m_ambient=3)


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


class KendallDisk:
    """Class used to plot points in Kendall shape space of 3D triangles.

    The shape space of 2D triangles is isometric to the 2-sphere of radius 1/2
    [K1984]. This isometry induced another isometry between the shape space of
    3D triangle and the 1-ball of radius pi/4 [LK1993]. Following the first
    visualization class "KendallSphere" for 2D triangles, this class encodes
    the 2D isometric representation of Kendall shape space of order (3,3).

    Attributes
    ----------
    points : list
        List of points to plot on the Kendall sphere.
    point_type : str
        Type of the points. Can be either 'pre-shape' (for points in Kendall
        pre-shape space) or 'extrinsic' (for points given as 3x2 matrices).
        Optional, default: 'pre-shape'.
    pole : array-like, shape=[3,2]
        Equilateral triangle in 2D (north pole).
    centre : array-like, shape=[3,3]
        Equilateral triangle in 3D (centre).
    ua : array-like, shape=[3,2]
        Tangent vector at north pole toward isocele triangle at vertex A.
    ub : array-like, shape=[3,2]
        Tangent vector at north pole toward isocele triangle at vertex B.
    na : array-like, shape=[3,2]
        Tangent vector such that (ua,na) is a positively oriented
        orthonormal basis of the horizontal space at north pole.

    References
    ----------
    .. [K1984] David G. Kendall. "Shape Manifolds, Procrustean Metrics, and
       Complex Projective Spaces." Bulletin of the London Mathematical
       Society, Volume 16, Issue 2, March 1984, Pages 81–121.
       https://doi.org/10.1112/blms/16.2.81
    .. [LK1993] Huiling Le and David G. Kendall. "The Riemannian structure of
       Euclidean shape spaces: a novel environment for statistics." Annals of
       statistics, 1993, vol. 21, no 3, p. 1225-1271.
       https://doi.org/10.1112/blms/16.2.81
    """

    def __init__(self, points=None, point_type="pre-shape"):
        self.points = []
        self.point_type = point_type
        self.ax = None

        self.pole = gs.array(
            [[1.0, 0.0], [-0.5, gs.sqrt(3.0) / 2.0], [-0.5, -gs.sqrt(3.0) / 2.0]]
        ) / gs.sqrt(3.0)

        self.centre = gs.array(
            [
                [1.0, 0.0, 0.0],
                [-0.5, gs.sqrt(3.0) / 2.0, 0.0],
                [-0.5, -gs.sqrt(3.0) / 2.0, 0.0],
            ]
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
            ax = plt.subplot()

        ax_s = gs.pi / 4 + 0.05
        plt.setp(ax, xlim=(-ax_s, ax_s), ylim=(-ax_s, ax_s), xlabel="X", ylabel="Y")
        self.ax = ax

    def convert_to_polar_coordinates(self, points):
        """Assign polar coordinates to given pre-shapes."""
        aligned_points = S33.align(points, self.centre)
        aligned_points2d = aligned_points[..., :, :2]
        speeds = S32.ambient_metric.log(aligned_points2d, self.pole)

        coords_r = S32.ambient_metric.dist(self.pole, aligned_points2d)
        coords_theta = gs.arctan2(
            S32.ambient_metric.inner_product(speeds, self.na),
            S32.ambient_metric.inner_product(speeds, self.ua),
        )

        return coords_r, coords_theta

    def convert_to_planar_coordinates(self, points):
        """Convert polar coordinates to spherical one."""
        coords_r, coords_theta = self.convert_to_polar_coordinates(points)
        coords_x = coords_r * gs.cos(coords_theta)
        coords_y = coords_r * gs.sin(coords_theta)
        planar_coords = gs.transpose(gs.stack((coords_x, coords_y)))
        return planar_coords

    def add_points(self, points):
        """Add points to draw on the Kendall disk."""
        if self.point_type == "extrinsic":
            if not gs.all(M33.belongs(points)):
                raise ValueError("Points do not belong to Matrices(3, 3).")
            points = S33.projection(points)
        elif self.point_type == "pre-shape" and not gs.all(S33.belongs(points)):
            raise ValueError("Points do not belong to the pre-shape space.")
        points = self.convert_to_planar_coordinates(points)
        if not isinstance(points, list):
            if points.shape == (2,):
                points = [gs.array(points)]
            else:
                points = list(points)
        self.points.extend(points)

    def clear_points(self):
        """Clear the points to draw."""
        self.points = []

    def draw(self, n_r=7, n_theta=25, scale=0.05):
        """Draw the disk regularly sampled with corresponding triangles."""
        self.set_ax()
        self.ax.set_axis_off()
        plt.tight_layout()

        coords_r = gs.linspace(0.0, gs.pi / 4.0, n_r)
        coords_theta = gs.linspace(0.0, 2.0 * gs.pi, n_theta)

        coords_x = gs.to_numpy(gs.outer(coords_r, gs.cos(coords_theta)))
        coords_y = gs.to_numpy(gs.outer(coords_r, gs.sin(coords_theta)))

        self.ax.fill(
            list(coords_x[-1, :]),
            list(coords_y[-1, :]),
            color="grey",
            alpha=0.1,
            zorder=-1,
        )
        for i_r in range(n_r):
            self.ax.plot(
                coords_x[i_r, :],
                coords_y[i_r, :],
                linewidth=0.6,
                color="grey",
                alpha=0.6,
                zorder=-1,
            )
        for i_t in range(n_theta):
            self.ax.plot(
                coords_x[:, i_t],
                coords_y[:, i_t],
                linewidth=0.6,
                color="grey",
                alpha=0.6,
                zorder=-1,
            )

        for r in gs.linspace(0.0, gs.pi / 4, n_r):
            for theta in gs.linspace(0.0, 2.0 * gs.pi, n_theta // 2 + 1):
                if theta == 0.0:
                    self.draw_triangle(0.0, 0.0, scale)
                else:
                    self.draw_triangle(r, theta, scale)

    def draw_triangle(self, r, theta, scale):
        """Draw the corresponding triangle on the disk at r, theta."""
        u_theta = gs.cos(theta) * self.ua + gs.sin(theta) * self.na
        triangle = gs.cos(r) * self.pole + gs.sin(r) * u_theta
        triangle = scale * triangle

        x = list(r * gs.cos(theta) + triangle[:, 0])
        x = x + [x[0]]
        y = list(r * gs.sin(theta) + triangle[:, 1])
        y = y + [y[0]]

        self.ax.plot(x, y, "grey", zorder=1)
        c = ["red", "green", "blue"]
        for i in range(3):
            self.ax.scatter(x[i], y[i], color=c[i], s=10, alpha=1, zorder=1)

    def draw_points(self, alpha=1, zorder=0, **kwargs):
        """Draw points on the Kendall disk."""
        points_x = [gs.to_numpy(point)[0] for point in self.points]
        points_y = [gs.to_numpy(point)[1] for point in self.points]
        self.ax.scatter(points_x, points_y, alpha=alpha, zorder=zorder, **kwargs)

    def draw_curve(self, alpha=1, zorder=0, **kwargs):
        """Draw a curve on the Kendall disk."""
        points_x = [gs.to_numpy(point)[0] for point in self.points]
        points_y = [gs.to_numpy(point)[1] for point in self.points]
        self.ax.plot(points_x, points_y, alpha=alpha, zorder=zorder, **kwargs)

    def draw_vector(self, tangent_vec, base_point, tol=1e-03, **kwargs):
        """Draw one vector in the tangent space to disk at a base point."""
        r_bp, th_bp = self.convert_to_polar_coordinates(base_point)
        bp = gs.array(
            [
                gs.cos(th_bp) * gs.sin(2 * r_bp),
                gs.sin(th_bp) * gs.sin(2 * r_bp),
                gs.cos(2 * r_bp),
            ]
        )
        r_exp, th_exp = self.convert_to_polar_coordinates(
            METRIC_S33.exp(
                tol * tangent_vec / METRIC_S33.norm(tangent_vec, base_point), base_point
            )
        )
        exp = gs.array(
            [
                gs.cos(th_exp) * gs.sin(2 * r_exp),
                gs.sin(th_exp) * gs.sin(2 * r_exp),
                gs.cos(2 * r_exp),
            ]
        )
        pole = gs.array([0.0, 0.0, 1.0])

        tv = exp - gs.dot(exp, bp) * bp
        u_tv = tv / gs.linalg.norm(tv)
        u_r = (gs.dot(pole, bp) * bp - pole) / gs.linalg.norm(
            gs.dot(pole, bp) * bp - pole
        )
        u_th = gs.cross(bp, u_r)
        x_r, x_th = gs.dot(u_tv, u_r), gs.dot(u_tv, u_th)

        bp = self.convert_to_planar_coordinates(base_point)
        u_r = bp / gs.linalg.norm(bp)
        u_th = gs.array([[0.0, -1.0], [1.0, 0.0]]) @ u_r
        tv = METRIC_S33.norm(tangent_vec, base_point) * (x_r * u_r + x_th * u_th)

        self.ax.quiver(bp[0], bp[1], tv[0], tv[1], **kwargs)
