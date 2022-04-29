"""Visualization for Geometric Statistics."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.pre_shape import KendallShapeMetric, PreShapeSpace
from geomstats.visualization._plotting import Plotter

S32 = PreShapeSpace(k_landmarks=3, m_ambient=2)

DEG2RAD = gs.pi / 180.0


class KendallSphere(Plotter):
    """Class used to plot points in Kendall shape space of 2D triangles.

    David G. Kendall showed that the shape space of 2D triangles is isometric
    to the 2-sphere of radius 1/2 [K1984]. This class encodes this isometric
    representation, offering a 3D visualization of Kendall shape space of order
    (3,2), and its related objects.

    Attributes
    ----------
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

    def __init__(self, point_type="pre-shape"):
        super().__init__()

        self.point_type = point_type

        self.space = PreShapeSpace(k_landmarks=3, m_ambient=2)
        self.metric = KendallShapeMetric(k_landmarks=3, m_ambient=2)
        if self.point_type == "extrinsic":
            self._matrices_space = Matrices(m=3, n=2)
            self._belongs = self._matrices_space.belongs
            self._project = self.space.projection
        elif self.point_type == "pre_shape":
            self._belongs = self.space.belongs

        self._convert_points = self._convert_to_spherical_coordinates

        self.pole, self.ua, self.ub, self.na = _init_shared_attrs()

        self._ax_scale = 0.5
        self._dim = 3

        _defaults = {
            "alpha": 1,
            "zorder": 0,
        }
        self._graph_defaults["scatter"] = _defaults
        self._graph_defaults["plot"] = _defaults

    def _set_view(self, ax, elev=60.0, azim=0.0):
        """Set azimuth and elevation angle."""
        ax.view_init(elev, azim)

        return ax

    def set_ax(self, ax=None, elev=60.0, azim=0.0):
        if ax is not None:
            return ax

        ax = super().set_ax(ax=ax)
        return self._set_view(ax, elev=elev, azim=azim)

    def _convert_to_polar_coordinates(self, points):
        """Assign polar coordinates to given pre-shapes."""
        aligned_points = self.space.align(points, self.pole)
        speeds = self.space.ambient_metric.log(aligned_points, self.pole)

        coords_theta = gs.arctan2(
            self.space.ambient_metric.inner_product(speeds, self.na),
            self.space.ambient_metric.inner_product(speeds, self.ua),
        )
        coords_phi = 2.0 * self.space.ambient_metric.dist(self.pole, aligned_points)

        return coords_theta, coords_phi

    def _convert_to_spherical_coordinates(self, points):
        """Convert polar coordinates to spherical one."""
        coords_theta, coords_phi = self._convert_to_polar_coordinates(points)
        coords_x = 0.5 * gs.cos(coords_theta) * gs.sin(coords_phi)
        coords_y = 0.5 * gs.sin(coords_theta) * gs.sin(coords_phi)
        coords_z = 0.5 * gs.cos(coords_phi)
        spherical_coords = gs.transpose(gs.stack((coords_x, coords_y, coords_z)))
        return spherical_coords

    @staticmethod
    def _rotate(theta, phi):
        """_rotate sending a triangle at pole to location theta, phi."""
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

    def _draw_triangle(self, ax, theta, phi, scale):
        """Draw the corresponding triangle on the sphere at theta, phi."""
        u_theta = gs.cos(theta) * self.ua + gs.sin(theta) * self.na
        triangle = gs.cos(phi / 2.0) * self.pole + gs.sin(phi / 2.0) * u_theta
        triangle = scale * triangle
        triangle3d = gs.transpose(
            gs.stack((triangle[:, 0], triangle[:, 1], 0.5 * gs.ones(3)))
        )
        triangle3d = self._rotate(theta, phi) @ gs.transpose(triangle3d)

        x = list(triangle3d[0]) + [triangle3d[0, 0]]
        y = list(triangle3d[1]) + [triangle3d[1, 0]]
        z = list(triangle3d[2]) + [triangle3d[2, 0]]

        ax.plot3D(x, y, z, "grey", zorder=1)
        c = ["red", "green", "blue"]
        for i in range(3):
            ax.scatter(x[i], y[i], z[i], color=c[i], s=10, alpha=1, zorder=1)

        return ax

    def scatter(
        self, points, ax=None, space_on=False, elev=60.0, azim=0.0, **scatter_kwargs
    ):
        ax_kwargs = {"elev": elev, "azim": azim}

        return super().scatter(
            points, ax=ax, space_on=space_on, ax_kwargs=ax_kwargs, **scatter_kwargs
        )

    def plot(self, points, ax=None, space_on=False, elev=60.0, azim=0.0, **plot_kwargs):
        ax_kwargs = {"elev": elev, "azim": azim}

        return super().plot(
            points, ax=ax, space_on=space_on, ax_kwargs=ax_kwargs, **plot_kwargs
        )

    def plot_inhabitants(
        self, ax=None, n_theta=25, n_phi=13, scale=0.05, elev=60.0, azim=0.0
    ):
        """Draw the sphere regularly sampled with corresponding triangles."""
        ax = self.plot_space(ax=ax, elev=elev, azim=azim)

        _elev, _azim = ax.elev * DEG2RAD, ax.azim * DEG2RAD

        def lim(theta):
            return gs.pi - _elev + (2.0 * _elev - gs.pi) / gs.pi * abs(_azim - theta)

        for theta in gs.linspace(0.0, 2.0 * gs.pi, n_theta // 2 + 1):
            for phi in gs.linspace(0.0, gs.pi, n_phi):
                if theta <= _azim + gs.pi and phi <= lim(theta):
                    self._draw_triangle(ax, theta, phi, scale)
                if theta > _azim + gs.pi and phi < lim(
                    2.0 * _azim + 2.0 * gs.pi - theta
                ):
                    self._draw_triangle(ax, theta, phi, scale)

        return ax

    def plot_space(self, ax=None, n_theta=25, n_phi=13, elev=60.0, azim=0.0):
        ax = self.set_ax(ax=ax, elev=elev, azim=azim)

        ax.set_axis_off()
        plt.tight_layout()

        coords_theta = gs.linspace(0.0, 2.0 * gs.pi, n_theta)
        coords_phi = gs.linspace(0.0, gs.pi, n_phi)

        coords_x = gs.to_numpy(0.5 * gs.outer(gs.sin(coords_phi), gs.cos(coords_theta)))
        coords_y = gs.to_numpy(0.5 * gs.outer(gs.sin(coords_phi), gs.sin(coords_theta)))
        coords_z = gs.to_numpy(
            0.5 * gs.outer(gs.cos(coords_phi), gs.ones_like(coords_theta))
        )

        ax.plot_surface(
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
        ax.plot_wireframe(
            coords_x,
            coords_y,
            coords_z,
            linewidths=0.6,
            color="grey",
            alpha=0.6,
            zorder=-1,
        )

        return ax

    def plot_geodesic(
        self,
        initial_point,
        end_point=None,
        initial_tangent_vec=None,
        n_points=1000,
        ax=None,
        elev=60.0,
        azim=0.0,
        space_on=False,
        **plot_kwargs
    ):
        """Plot geodesic.

        Follows metric.geodesic signature.
        """
        ax_kwargs = {"elev": elev, "azim": azim}

        return super().plot_geodesic(
            initial_point,
            end_point=end_point,
            initial_tangent_vec=initial_tangent_vec,
            n_points=n_points,
            ax=ax,
            space_on=space_on,
            ax_kwargs=ax_kwargs,
            **plot_kwargs
        )

    def plot_vector_field(
        self,
        tangent_vec,
        base_point,
        ax=None,
        elev=60.0,
        azim=0.0,
        space_on=False,
        **quiver_kwargs
    ):
        """Draw vectors in the tangent space to sphere at specific base points."""

        ax_kwargs = {"elev": elev, "azim": azim}
        ax, _ = self._prepare_vis(
            ax,
            None,
            space_on=space_on,
            grid_on=False,
            ax_kwargs=ax_kwargs,
        )

        norm = self.metric.norm(tangent_vec, base_point)
        exp = self.metric.exp(tangent_vec, base_point)

        bp = self._convert_to_spherical_coordinates(base_point)
        exp = self._convert_to_spherical_coordinates(exp)

        tv = exp - _scalar_vec_mult(_dot(exp, 2.0 * bp), 2.0 * bp)
        tv = _normalize_vec(tv)
        tv = _scalar_vec_mult(norm, tv)

        ax.quiver(
            *[bp[..., i] for i in range(self._dim)],
            *[tv[..., i] for i in range(self._dim)],
            **quiver_kwargs
        )

        return ax


class KendallDisk(Plotter):
    """Class used to plot points in Kendall shape space of 3D triangles.

    The shape space of 2D triangles is isometric to the 2-sphere of radius 1/2
    [K1984]. This isometry induced another isometry between the shape space of
    3D triangle and the 1-ball of radius pi/4 [LK1993]. Following the first
    visualization class "KendallSphere" for 2D triangles, this class encodes
    the 2D isometric representation of Kendall shape space of order (3,3).

    Attributes
    ----------
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

    def __init__(self, point_type="pre-shape"):
        super().__init__()

        self.point_type = point_type

        self.space = PreShapeSpace(k_landmarks=3, m_ambient=3)
        self.metric = KendallShapeMetric(k_landmarks=3, m_ambient=3)
        if self.point_type == "extrinsic":
            self._matrices_space = Matrices(m=3, n=3)
            self._belongs = self._matrices_space.belongs
            self._project = self.space.projection
        elif self.point_type == "pre_shape":
            self._belongs = self.space.belongs

        self._convert_points = self._convert_to_planar_coordinates

        self.centre = gs.array(
            [
                [1.0, 0.0, 0.0],
                [-0.5, gs.sqrt(3.0) / 2.0, 0.0],
                [-0.5, -gs.sqrt(3.0) / 2.0, 0.0],
            ]
        ) / gs.sqrt(3.0)

        self.pole, self.ua, self.ub, self.na = _init_shared_attrs()

        self._ax_scale = gs.pi / 4.0 + 0.05
        self._dim = 2

    def _convert_to_polar_coordinates(self, points):
        """Assign polar coordinates to given pre-shapes."""
        aligned_points = self.space.align(points, self.centre)
        aligned_points2d = aligned_points[..., :, :2]
        speeds = S32.ambient_metric.log(aligned_points2d, self.pole)

        coords_r = S32.ambient_metric.dist(self.pole, aligned_points2d)
        coords_theta = gs.arctan2(
            S32.ambient_metric.inner_product(speeds, self.na),
            S32.ambient_metric.inner_product(speeds, self.ua),
        )

        return coords_r, coords_theta

    def _convert_to_planar_coordinates(self, points):
        """Convert polar coordinates to spherical one."""
        coords_r, coords_theta = self._convert_to_polar_coordinates(points)
        coords_x = coords_r * gs.cos(coords_theta)
        coords_y = coords_r * gs.sin(coords_theta)
        planar_coords = gs.transpose(gs.stack((coords_x, coords_y)))
        return planar_coords

    def _draw_triangle(self, ax, r, theta, scale):
        """Draw the corresponding triangle on the disk at r, theta."""
        u_theta = gs.cos(theta) * self.ua + gs.sin(theta) * self.na
        triangle = gs.cos(r) * self.pole + gs.sin(r) * u_theta
        triangle = scale * triangle

        x = list(r * gs.cos(theta) + triangle[:, 0])
        x = x + [x[0]]
        y = list(r * gs.sin(theta) + triangle[:, 1])
        y = y + [y[0]]

        ax.plot(x, y, "grey", zorder=1)
        c = ["red", "green", "blue"]
        for i in range(3):
            ax.scatter(x[i], y[i], color=c[i], s=10, alpha=1, zorder=1)

    def plot_inhabitants(self, ax=None, n_r=7, n_theta=25, scale=0.05):
        ax = self.plot_space(ax=ax, n_r=n_r, n_theta=n_theta, scale=scale)

        for r in gs.linspace(0.0, gs.pi / 4, n_r):
            for theta in gs.linspace(0.0, 2.0 * gs.pi, n_theta // 2 + 1):
                if theta == 0.0:
                    self._draw_triangle(ax, 0.0, 0.0, scale)
                else:
                    self._draw_triangle(ax, r, theta, scale)

        return ax

    def plot_space(self, ax=None, n_r=7, n_theta=25, scale=0.05):
        """Draw the disk regularly sampled with corresponding triangles."""
        ax = self.set_ax(ax=ax)

        ax.set_axis_off()
        plt.tight_layout()

        coords_r = gs.linspace(0.0, gs.pi / 4.0, n_r)
        coords_theta = gs.linspace(0.0, 2.0 * gs.pi, n_theta)

        coords_x = gs.to_numpy(gs.outer(coords_r, gs.cos(coords_theta)))
        coords_y = gs.to_numpy(gs.outer(coords_r, gs.sin(coords_theta)))

        ax.fill(
            list(coords_x[-1, :]),
            list(coords_y[-1, :]),
            color="grey",
            alpha=0.1,
            zorder=-1,
        )
        for i_r in range(n_r):
            ax.plot(
                coords_x[i_r, :],
                coords_y[i_r, :],
                linewidth=0.6,
                color="grey",
                alpha=0.6,
                zorder=-1,
            )
        for i_t in range(n_theta):
            ax.plot(
                coords_x[:, i_t],
                coords_y[:, i_t],
                linewidth=0.6,
                color="grey",
                alpha=0.6,
                zorder=-1,
            )

        return ax

    def plot_vector_field(
        self,
        tangent_vec,
        base_point,
        ax=None,
        space_on=False,
        tol=1e-03,
        **quiver_kwargs
    ):
        """Draw vectors in the tangent space to sphere at specific base points."""

        ax, _ = self._prepare_vis(ax, None, space_on=space_on, grid_on=False)

        r_bp, th_bp = self._convert_to_polar_coordinates(base_point)
        bp = gs.array(
            [
                gs.cos(th_bp) * gs.sin(2 * r_bp),
                gs.sin(th_bp) * gs.sin(2 * r_bp),
                gs.cos(2 * r_bp),
            ]
        ).T

        r_exp, th_exp = self._convert_to_polar_coordinates(
            self.metric.exp(
                _scalar_mat_mult(
                    1.0 / self.metric.norm(tangent_vec, base_point), tol * tangent_vec
                ),
                base_point,
            )
        )

        exp = gs.array(
            [
                gs.cos(th_exp) * gs.sin(2 * r_exp),
                gs.sin(th_exp) * gs.sin(2 * r_exp),
                gs.cos(2 * r_exp),
            ]
        ).T

        pole = gs.array([0.0, 0.0, 1.0])

        tv = exp - _scalar_vec_mult(_dot(exp, bp), bp)
        u_tv = _normalize_vec(tv)

        u_r = _scalar_vec_mult(_dot(pole, bp), bp) - pole
        u_r = _normalize_vec(u_r)

        u_th = gs.cross(bp, u_r, axis=-1)

        x_r, x_th = _dot(u_tv, u_r), _dot(u_tv, u_th)

        bp = self._convert_to_planar_coordinates(base_point)
        u_r = _normalize_vec(bp)

        u_th = gs.einsum("ij,...j->...i", gs.array([[0.0, -1.0], [1.0, 0.0]]), u_r)

        tv = _scalar_vec_mult(
            self.metric.norm(tangent_vec, base_point),
            _scalar_vec_mult(x_r, u_r) + _scalar_vec_mult(x_th, u_th),
        )

        ax.quiver(
            *[bp[..., i] for i in range(self._dim)],
            *[tv[..., i] for i in range(self._dim)],
            **quiver_kwargs
        )

        return ax


def _init_shared_attrs():

    pole = gs.array(
        [[1.0, 0.0], [-0.5, gs.sqrt(3.0) / 2.0], [-0.5, -gs.sqrt(3.0) / 2.0]]
    ) / gs.sqrt(3.0)

    ua = gs.array(
        [[-1.0, 0.0], [0.5, gs.sqrt(3.0) / 2.0], [0.5, -gs.sqrt(3.0) / 2.0]]
    ) / gs.sqrt(3.0)

    ub = gs.array(
        [[0.5, gs.sqrt(3.0) / 2.0], [0.5, -gs.sqrt(3.0) / 2], [-1.0, 0.0]]
    ) / gs.sqrt(3.0)

    na = ub - S32.ambient_metric.inner_product(ub, ua) * ua
    na = na / S32.ambient_metric.norm(na)

    return pole, ua, ub, na


def _scalar_mat_mult(scalar, vector):
    return gs.einsum("...,...ij->...ij", scalar, vector)


def _scalar_vec_mult(scalar, vector):
    return gs.einsum("...,...i->...i", scalar, vector)


def _dot(vec1, vec2):
    return gs.einsum("...i,...i->...", vec1, vec2)


def _normalize_vec(vec):
    return _scalar_vec_mult(1.0 / gs.linalg.norm(vec, axis=-1), vec)
