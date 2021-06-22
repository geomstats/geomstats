"""Visualization for Geometric Statistics."""
import logging

import matplotlib
import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.pre_shape import KendallShapeMetric, PreShapeSpace
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from mpl_toolkits.mplot3d import Axes3D  # NOQA

SE3_GROUP = SpecialEuclidean(n=3, point_type='vector')
SE2_GROUP = SpecialEuclidean(n=2, point_type='matrix')
SE2_VECT = SpecialEuclidean(n=2, point_type='vector')
SO3_GROUP = SpecialOrthogonal(n=3, point_type='vector')
S1 = Hypersphere(dim=1)
S2 = Hypersphere(dim=2)
H2 = Hyperboloid(dim=2)
POINCARE_HALF_PLANE = PoincareHalfSpace(dim=2)
M32 = Matrices(m=3, n=2)
S32 = PreShapeSpace(k_landmarks=3, m_ambient=2)
METRIC_S32 = KendallShapeMetric(k_landmarks=3, m_ambient=2)
M33 = Matrices(m=3, n=3)
S33 = PreShapeSpace(k_landmarks=3, m_ambient=3)
METRIC_S33 = KendallShapeMetric(k_landmarks=3, m_ambient=3)

AX_SCALE = 1.2

IMPLEMENTED = ['SO3_GROUP', 'SE3_GROUP', 'SE2_GROUP', 'S1', 'S2',
               'H2_poincare_disk', 'H2_poincare_half_plane', 'H2_klein_disk',
               'poincare_polydisk', 'S32', 'M32', 'S33', 'M33']


def tutorial_matplotlib():
    fontsize = 12
    matplotlib.rc('font', size=fontsize)
    matplotlib.rc('text')
    matplotlib.rc('legend', fontsize=fontsize)
    matplotlib.rc('axes', titlesize=21, labelsize=14)
    matplotlib.rc(
        'font',
        family='times',
        serif=['Computer Modern Roman'],
        monospace=['Computer Modern Typewriter'])


class Arrow3D:
    """An arrow in 3d, i.e. a point and a vector."""

    def __init__(self, point, vector):
        self.point = point
        self.vector = vector

    def draw(self, ax, **quiver_kwargs):
        """Draw the arrow in 3D plot."""
        ax.quiver(self.point[0], self.point[1], self.point[2],
                  self.vector[0], self.vector[1], self.vector[2],
                  **quiver_kwargs)


class Trihedron:
    """A trihedron, i.e. 3 Arrow3Ds at the same point."""

    def __init__(self, point, vec_1, vec_2, vec_3):
        self.arrow_1 = Arrow3D(point, vec_1)
        self.arrow_2 = Arrow3D(point, vec_2)
        self.arrow_3 = Arrow3D(point, vec_3)

    def draw(self, ax, **arrow_draw_kwargs):
        """Draw the trihedron by drawing its 3 Arrow3Ds.

        Arrows are drawn is order using green, red, and blue
        to show the trihedron's orientation.
        """
        if 'color' in arrow_draw_kwargs:
            self.arrow_1.draw(ax, **arrow_draw_kwargs)
            self.arrow_2.draw(ax, **arrow_draw_kwargs)
            self.arrow_3.draw(ax, **arrow_draw_kwargs)
        else:
            blue = '#1f77b4'
            orange = '#ff7f0e'
            green = '#2ca02c'
            self.arrow_1.draw(ax, color=blue, **arrow_draw_kwargs)
            self.arrow_2.draw(ax, color=orange, **arrow_draw_kwargs)
            self.arrow_3.draw(ax, color=green, **arrow_draw_kwargs)


class Circle:
    """Class used to draw a circle."""

    def __init__(self, n_angles=100, points=None):
        angles = gs.linspace(0, 2 * gs.pi, n_angles)
        self.circle_x = gs.cos(angles)
        self.circle_y = gs.sin(angles)
        self.points = []
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None):
        if ax is None:
            ax = plt.subplot()
        ax_s = AX_SCALE
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(-ax_s, ax_s),
                 xlabel='X', ylabel='Y')
        return ax

    def add_points(self, points):
        if not gs.all(S1.belongs(points)):
            raise ValueError('Points do  not belong to the circle.')
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def draw(self, ax, **plot_kwargs):
        ax.plot(self.circle_x, self.circle_y, color="black")
        if self.points:
            self.draw_points(ax, **plot_kwargs)

    def draw_points(self, ax, points=None, **plot_kwargs):
        if points is None:
            points = self.points
        points = gs.array(points)
        ax.plot(points[:, 0], points[:, 1], marker='o', linestyle="None",
                **plot_kwargs)


class Sphere:
    """Create the arrays sphere_x, sphere_y, sphere_z to plot a sphere.

    Create the arrays sphere_x, sphere_y, sphere_z of values
    to plot the wireframe of a sphere.
    Their shape is (n_meridians, n_circles_latitude).
    """

    def __init__(self, n_meridians=40, n_circles_latitude=None,
                 points=None):
        if n_circles_latitude is None:
            n_circles_latitude = max(n_meridians / 2, 4)

        u, v = gs.meshgrid(
            gs.arange(0, 2 * gs.pi, 2 * gs.pi / n_meridians),
            gs.arange(0, gs.pi, gs.pi / n_circles_latitude))

        self.center = gs.zeros(3)
        self.radius = 1
        self.sphere_x = self.center[0] + self.radius * gs.cos(u) * gs.sin(v)
        self.sphere_y = self.center[1] + self.radius * gs.sin(u) * gs.sin(v)
        self.sphere_z = self.center[2] + self.radius * gs.cos(v)

        self.points = []
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None):
        if ax is None:
            ax = plt.subplot(111, projection='3d')

        ax_s = AX_SCALE
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(-ax_s, ax_s),
                 zlim=(-ax_s, ax_s),
                 xlabel='X', ylabel='Y', zlabel='Z')
        return ax

    def add_points(self, points):
        if not gs.all(S2.belongs(points)):
            raise ValueError('Points do not belong to the sphere.')
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def draw(self, ax, **scatter_kwargs):
        ax.plot_wireframe(self.sphere_x,
                          self.sphere_y,
                          self.sphere_z,
                          color="grey", alpha=0.2)
        ax.set_box_aspect([1., 1., 1.])
        if self.points:
            self.draw_points(ax, **scatter_kwargs)

    def draw_points(self, ax, points=None, **scatter_kwargs):
        if points is None:
            points = self.points
        points_x = [point[0] for point in points]
        points_y = [point[1] for point in points]
        points_z = [point[2] for point in points]
        ax.scatter(points_x, points_y, points_z, **scatter_kwargs)

        for i_point, point in enumerate(points):
            if 'label' in scatter_kwargs:
                if len(scatter_kwargs['label']) == len(points):
                    ax.text(
                        point[0], point[1], point[2],
                        scatter_kwargs['label'][i_point],
                        size=10, zorder=1, color='k')

    def fibonnaci_points(self, n_points=16000):
        """Spherical Fibonacci point sets yield nearly uniform point
        distributions on the unit sphere."""
        x_vals = []
        y_vals = []
        z_vals = []

        offset = 2. / n_points
        increment = gs.pi * (3. - gs.sqrt(5.))

        for i in range(n_points):
            y = ((i * offset) - 1) + (offset / 2)
            r = gs.sqrt(1 - pow(y, 2))

            phi = ((i + 1) % n_points) * increment

            x = gs.cos(phi) * r
            z = gs.sin(phi) * r

            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)

        x_vals = [(self.radius * i) for i in x_vals]
        y_vals = [(self.radius * i) for i in y_vals]
        z_vals = [(self.radius * i) for i in z_vals]

        return gs.array([x_vals, y_vals, z_vals])

    def plot_heatmap(self, ax,
                     scalar_function,
                     n_points=16000,
                     alpha=0.2,
                     cmap='jet'):
        """Plot a heatmap defined by a loss on the sphere."""
        points = self.fibonnaci_points(n_points)
        intensity = gs.array([scalar_function(x) for x in points.T])
        ax.scatter(points[0, :], points[1, :], points[2, :],
                   c=intensity,
                   alpha=alpha,
                   marker='.',
                   cmap=plt.get_cmap(cmap))


class PoincareDisk:
    def __init__(self, points=None, point_type='extrinsic'):
        self.center = gs.array([0., 0.])
        self.points = []
        self.point_type = point_type
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None):
        if ax is None:
            ax = plt.subplot()
        ax_s = AX_SCALE
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(-ax_s, ax_s),
                 xlabel='X', ylabel='Y')
        return ax

    def add_points(self, points):

        if self.point_type == 'extrinsic':
            if not gs.all(H2.belongs(points)):
                raise ValueError(
                    'Points do not belong to the hyperbolic space.')
            points = self.convert_to_poincare_coordinates(points)

        if not isinstance(points, list):
            points = list(points)

        if gs.all([len(point) == 2 for point in self.points]):
            self.points.extend(points)
        else:
            raise ValueError('Points do not have dimension 2.')

    @staticmethod
    def convert_to_poincare_coordinates(points):
        poincare_coords = points[:, 1:] / (1 + points[:, :1])
        return poincare_coords

    def draw(self, ax, **kwargs):
        circle = plt.Circle((0, 0), radius=1., color='black', fill=False)
        ax.add_artist(circle)
        if len(self.points) > 0:
            if gs.all([len(point) == 2 for point in self.points]):
                points_x = gs.stack(
                    [point[0] for point in self.points], axis=0)
                points_y = gs.stack(
                    [point[1] for point in self.points], axis=0)
                ax.scatter(points_x, points_y, **kwargs)
            else:
                raise ValueError('Points do not have dimension 2.')


class PoincarePolyDisk:
    """Class used to plot points in the Poincare polydisk."""

    def __init__(self, points=None, point_type='ball', n_disks=2):
        self.center = gs.array([0., 0.])
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
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(-ax_s, ax_s),
                 xlabel='X', ylabel='Y')
        return ax

    def add_points(self, points):
        """Add points to draw."""
        if self.point_type == 'extrinsic':
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
        circle = plt.Circle((0, 0), radius=1., color='black', fill=False)
        ax.add_artist(circle)
        points_x = [gs.to_numpy(point[0]) for point in self.points]
        points_y = [gs.to_numpy(point[1]) for point in self.points]
        ax.scatter(points_x, points_y, **kwargs)


class PoincareHalfPlane:
    """Class used to plot points in the Poincare Half Plane."""

    def __init__(self, points=None, point_type='half-space'):
        self.points = []
        self.point_type = point_type
        if points is not None:
            self.add_points(points)

    def add_points(self, points):
        if self.point_type == 'extrinsic':
            if not gs.all(H2.belongs(points)):
                raise ValueError(
                    'Points do not belong to the hyperbolic space '
                    '(extrinsic coordinates)')
            points = self.convert_to_half_plane_coordinates(points)
        elif self.point_type == 'half-space':
            if not gs.all(POINCARE_HALF_PLANE.belongs(points)):
                raise ValueError(
                    'Points do not belong to the hyperbolic space '
                    '(Poincare half plane coordinates).')
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def set_ax(self, points, ax=None):
        if ax is None:
            ax = plt.subplot()
        if self.point_type == 'extrinsic':
            points = self.convert_to_half_plane_coordinates(points)
        plt.setp(ax, xlabel='X', ylabel='Y')
        return ax

    @staticmethod
    def convert_to_half_plane_coordinates(points):
        disk_coords = points[:, 1:] / (1 + points[:, :1])
        disk_x = disk_coords[:, 0]
        disk_y = disk_coords[:, 1]

        denominator = (disk_x ** 2 + (1 - disk_y) ** 2)
        coords_0 = gs.expand_dims(2 * disk_x / denominator, axis=1)
        coords_1 = gs.expand_dims(
            (1 - disk_x ** 2 - disk_y ** 2) / denominator, axis=1)

        half_plane_coords = gs.concatenate(
            [coords_0, coords_1], axis=1)
        return half_plane_coords

    def draw(self, ax, **kwargs):
        points_x = [gs.to_numpy(point[0]) for point in self.points]
        points_y = [gs.to_numpy(point[1]) for point in self.points]
        ax.scatter(points_x, points_y, **kwargs)


class KleinDisk:
    def __init__(self, points=None):
        self.center = gs.array([0., 0.])
        self.points = []
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None):
        if ax is None:
            ax = plt.subplot()
        ax_s = AX_SCALE
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(-ax_s, ax_s),
                 xlabel='X', ylabel='Y')
        return ax

    def add_points(self, points):
        if not gs.all(H2.belongs(points)):
            raise ValueError(
                'Points do not belong to the hyperbolic space.')
        points = self.convert_to_klein_coordinates(points)
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    @staticmethod
    def convert_to_klein_coordinates(points):
        poincare_coords = points[:, 1:] / (1 + points[:, :1])
        poincare_radius = gs.linalg.norm(
            poincare_coords, axis=1)
        poincare_angle = gs.arctan2(
            poincare_coords[:, 1], poincare_coords[:, 0])

        klein_radius = 2 * poincare_radius / (1 + poincare_radius ** 2)
        klein_angle = poincare_angle

        coords_0 = gs.expand_dims(
            klein_radius * gs.cos(klein_angle), axis=1)
        coords_1 = gs.expand_dims(
            klein_radius * gs.sin(klein_angle), axis=1)
        klein_coords = gs.concatenate([coords_0, coords_1], axis=1)
        return klein_coords

    def draw(self, ax, **kwargs):
        circle = plt.Circle((0, 0), radius=1., color='black', fill=False)
        ax.add_artist(circle)
        points_x = [gs.to_numpy(point[0]) for point in self.points]
        points_y = [gs.to_numpy(point[1]) for point in self.points]
        ax.scatter(points_x, points_y, **kwargs)


class SpecialEuclidean2:
    """Class used to plot points in the 2d special euclidean group."""

    def __init__(self, points=None, point_type='matrix'):
        self.points = []
        self.point_type = point_type
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None, x_lim=None, y_lim=None):
        if ax is None:
            ax = plt.subplot()
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        return ax

    def add_points(self, points):
        if self.point_type == 'vector':
            points = SE2_VECT.matrix_from_vector(points)
        if not gs.all(SE2_GROUP.belongs(points)):
            logging.warning(
                'Some points do not belong to SE2.')
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def draw(self, ax, **kwargs):
        points = gs.array(self.points)
        translation = points[..., :2, 2]
        frame_1 = points[:, :2, 0]
        frame_2 = points[:, :2, 1]
        ax.quiver(translation[:, 0], translation[:, 1], frame_1[:, 0],
                  frame_1[:, 1], width=0.005, color='b')
        ax.quiver(translation[:, 0], translation[:, 1], frame_2[:, 0],
                  frame_2[:, 1], width=0.005, color='r')
        ax.scatter(translation[:, 0], translation[:, 1], s=16, **kwargs)


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

    def __init__(self, points=None, point_type='pre-shape'):
        self.points = []
        self.point_type = point_type
        self.ax = None
        self.elev, self.azim = None, None

        self.pole = gs.array([[1., 0.],
                              [-.5, gs.sqrt(3.) / 2.],
                              [-.5, -gs.sqrt(3.) / 2.]]) / gs.sqrt(3.)

        self.ua = gs.array([[-1., 0.],
                            [.5, gs.sqrt(3.) / 2.],
                            [.5, -gs.sqrt(3.) / 2.]]) / gs.sqrt(3.)

        self.ub = gs.array([[.5, gs.sqrt(3.) / 2.],
                            [.5, -gs.sqrt(3.) / 2],
                            [-1., 0.]]) / gs.sqrt(3.)

        self.na = self.ub - S32.ambient_metric.inner_product(
            self.ub, self.ua) * self.ua
        self.na = self.na / S32.ambient_metric.norm(self.na)

        if points is not None:
            self.add_points(points)

    def set_ax(self, ax=None):
        """Set axis."""
        if ax is None:
            ax = plt.subplot(111, projection='3d')

        ax_s = .5
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(-ax_s, ax_s),
                 zlim=(-ax_s, ax_s),
                 xlabel='X', ylabel='Y', zlabel='Z')
        self.ax = ax

    def set_view(self, elev=60., azim=0.):
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
            S32.ambient_metric.inner_product(speeds, self.ua))
        coords_phi = 2. * S32.ambient_metric.dist(self.pole, aligned_points)

        return coords_theta, coords_phi

    def convert_to_spherical_coordinates(self, points):
        """Convert polar coordinates to spherical one."""
        coords_theta, coords_phi = self.convert_to_polar_coordinates(points)
        coords_x = .5 * gs.cos(coords_theta) * gs.sin(coords_phi)
        coords_y = .5 * gs.sin(coords_theta) * gs.sin(coords_phi)
        coords_z = .5 * gs.cos(coords_phi)
        spherical_coords = gs.transpose(gs.stack(
            (coords_x, coords_y, coords_z)))
        return spherical_coords

    def add_points(self, points):
        """Add points to draw on the Kendall sphere."""
        if self.point_type == 'extrinsic':
            if not gs.all(M32.belongs(points)):
                raise ValueError(
                    'Points do not belong to Matrices(3, 2).')
            points = S32.projection(points)
        elif self.point_type == 'pre-shape' \
                and not gs.all(S32.belongs(points)):
            raise ValueError('Points do not belong to the pre-shape space.')
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

    def draw(self, n_theta=25, n_phi=13, scale=.05, elev=60., azim=0.):
        """Draw the sphere regularly sampled with corresponding triangles."""
        self.set_ax()
        self.set_view(elev=elev, azim=azim)
        self.ax.set_axis_off()
        plt.tight_layout()

        coords_theta = gs.linspace(0., 2. * gs.pi, n_theta)
        coords_phi = gs.linspace(0., gs.pi, n_phi)

        coords_x = gs.to_numpy(.5 * gs.outer(gs.sin(coords_phi),
                                             gs.cos(coords_theta)))
        coords_y = gs.to_numpy(.5 * gs.outer(gs.sin(coords_phi),
                                             gs.sin(coords_theta)))
        coords_z = gs.to_numpy(.5 * gs.outer(gs.cos(coords_phi),
                                             gs.ones_like(coords_theta)))

        self.ax.plot_surface(coords_x, coords_y, coords_z,
                             rstride=1, cstride=1, color='grey',
                             linewidth=0, alpha=.1, zorder=-1)
        self.ax.plot_wireframe(coords_x, coords_y, coords_z, linewidths=.6,
                               color='grey', alpha=.6, zorder=-1)

        def lim(theta):
            return gs.pi - self.elev + (2. * self.elev - gs.pi) / gs.pi * abs(
                self.azim - theta)

        for theta in gs.linspace(0., 2. * gs.pi, n_theta // 2 + 1):
            for phi in gs.linspace(0., gs.pi, n_phi):
                if theta <= self.azim + gs.pi and phi <= lim(theta):
                    self.draw_triangle(theta, phi, scale)
                if theta > self.azim + gs.pi and phi \
                        < lim(2. * self.azim + 2. * gs.pi - theta):
                    self.draw_triangle(theta, phi, scale)

    def draw_triangle(self, theta, phi, scale):
        """Draw the corresponding triangle on the sphere at theta, phi."""
        u_theta = gs.cos(theta) * self.ua + gs.sin(theta) * self.na
        triangle = gs.cos(phi / 2.) * self.pole + gs.sin(phi / 2.) * u_theta
        triangle = scale * triangle
        triangle3d = gs.transpose(gs.stack((triangle[:, 0], triangle[:, 1],
                                            .5 * gs.ones(3))))
        triangle3d = self.rotation(theta, phi) @ gs.transpose(triangle3d)

        x = list(triangle3d[0]) + [triangle3d[0, 0]]
        y = list(triangle3d[1]) + [triangle3d[1, 0]]
        z = list(triangle3d[2]) + [triangle3d[2, 0]]

        self.ax.plot3D(x, y, z, 'grey', zorder=1)
        c = ['red', 'green', 'blue']
        for i in range(3):
            self.ax.scatter(x[i], y[i], z[i],
                            color=c[i], s=10, alpha=1, zorder=1)

    @staticmethod
    def rotation(theta, phi):
        """Rotation sending a triangle at pole to location theta, phi."""
        rot_th = gs.array([[gs.cos(theta), -gs.sin(theta), 0.],
                           [gs.sin(theta), gs.cos(theta), 0.],
                           [0., 0., 1.]])
        rot_phi = gs.array([[gs.cos(phi), 0., gs.sin(phi)],
                            [0., 1., 0.],
                            [-gs.sin(phi), 0, gs.cos(phi)]])
        return rot_th @ rot_phi @ gs.transpose(rot_th)

    def draw_points(self, alpha=1, zorder=0, **kwargs):
        """Draw points on the Kendall sphere."""
        points_x = [gs.to_numpy(point)[0] for point in self.points]
        points_y = [gs.to_numpy(point)[1] for point in self.points]
        points_z = [gs.to_numpy(point)[2] for point in self.points]
        self.ax.scatter(points_x, points_y, points_z,
                        alpha=alpha, zorder=zorder, **kwargs)

    def draw_curve(self, alpha=1, zorder=0, **kwargs):
        """Draw a curve on the Kendall sphere."""
        points_x = [gs.to_numpy(point)[0] for point in self.points]
        points_y = [gs.to_numpy(point)[1] for point in self.points]
        points_z = [gs.to_numpy(point)[2] for point in self.points]
        self.ax.plot3D(points_x, points_y, points_z,
                       alpha=alpha, zorder=zorder, **kwargs)

    def draw_vector(self, tangent_vec, base_point, **kwargs):
        """Draw one vector in the tangent space to sphere at a base point."""
        norm = METRIC_S32.norm(tangent_vec, base_point)
        exp = METRIC_S32.exp(tangent_vec, base_point)
        bp = self.convert_to_spherical_coordinates(base_point)
        exp = self.convert_to_spherical_coordinates(exp)
        tv = exp - gs.dot(exp, 2. * bp) * 2. * bp
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

    def __init__(self, points=None, point_type='pre-shape'):
        self.points = []
        self.point_type = point_type
        self.ax = None

        self.pole = gs.array([[1., 0.],
                              [-.5, gs.sqrt(3.) / 2.],
                              [-.5, -gs.sqrt(3.) / 2.]]) / gs.sqrt(3.)

        self.centre = gs.array([[1., 0., 0.],
                                [-.5, gs.sqrt(3.) / 2., 0.],
                                [-.5, -gs.sqrt(3.) / 2., 0.]]) / gs.sqrt(3.)

        self.ua = gs.array([[-1., 0.],
                            [.5, gs.sqrt(3.) / 2.],
                            [.5, -gs.sqrt(3.) / 2.]]) / gs.sqrt(3.)

        self.ub = gs.array([[.5, gs.sqrt(3.) / 2.],
                            [.5, -gs.sqrt(3.) / 2],
                            [-1., 0.]]) / gs.sqrt(3.)

        self.na = self.ub - S32.ambient_metric.inner_product(
            self.ub, self.ua) * self.ua
        self.na = self.na / S32.ambient_metric.norm(self.na)

        if points is not None:
            self.add_points(points)

    def set_ax(self, ax=None):
        """Set axis."""
        if ax is None:
            ax = plt.subplot()

        ax_s = gs.pi / 4 + .05
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(-ax_s, ax_s),
                 xlabel='X', ylabel='Y')
        self.ax = ax

    def convert_to_polar_coordinates(self, points):
        """Assign polar coordinates to given pre-shapes."""
        aligned_points = S33.align(points, self.centre)
        aligned_points2d = aligned_points[..., :, :2]
        speeds = S32.ambient_metric.log(aligned_points2d, self.pole)

        coords_r = S32.ambient_metric.dist(self.pole, aligned_points2d)
        coords_theta = gs.arctan2(
            S32.ambient_metric.inner_product(speeds, self.na),
            S32.ambient_metric.inner_product(speeds, self.ua))

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
        if self.point_type == 'extrinsic':
            if not gs.all(M33.belongs(points)):
                raise ValueError(
                    'Points do not belong to Matrices(3, 3).')
            points = S33.projection(points)
        elif self.point_type == 'pre-shape' \
                and not gs.all(S33.belongs(points)):
            raise ValueError('Points do not belong to the pre-shape space.')
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

    def draw(self, n_r=7, n_theta=25, scale=.05):
        """Draw the disk regularly sampled with corresponding triangles."""
        self.set_ax()
        self.ax.set_axis_off()
        plt.tight_layout()

        coords_r = gs.linspace(0., gs.pi / 4., n_r)
        coords_theta = gs.linspace(0., 2. * gs.pi, n_theta)

        coords_x = gs.to_numpy(gs.outer(coords_r, gs.cos(coords_theta)))
        coords_y = gs.to_numpy(gs.outer(coords_r, gs.sin(coords_theta)))

        self.ax.fill(list(coords_x[-1, :]), list(coords_y[-1, :]),
                     color='grey', alpha=.1, zorder=-1)
        for i_r in range(n_r):
            self.ax.plot(coords_x[i_r, :], coords_y[i_r, :],
                         linewidth=.6, color='grey', alpha=.6, zorder=-1)
        for i_t in range(n_theta):
            self.ax.plot(coords_x[:, i_t], coords_y[:, i_t],
                         linewidth=.6, color='grey', alpha=.6, zorder=-1)

        for r in gs.linspace(0., gs.pi / 4, n_r):
            for theta in gs.linspace(0., 2. * gs.pi, n_theta // 2 + 1):
                if theta == 0.:
                    self.draw_triangle(0., 0., scale)
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

        self.ax.plot(x, y, 'grey', zorder=1)
        c = ['red', 'green', 'blue']
        for i in range(3):
            self.ax.scatter(x[i], y[i],
                            color=c[i], s=10, alpha=1, zorder=1)

    def draw_points(self, alpha=1, zorder=0, **kwargs):
        """Draw points on the Kendall disk."""
        points_x = [gs.to_numpy(point)[0] for point in self.points]
        points_y = [gs.to_numpy(point)[1] for point in self.points]
        self.ax.scatter(points_x, points_y,
                        alpha=alpha, zorder=zorder, **kwargs)

    def draw_curve(self, alpha=1, zorder=0, **kwargs):
        """Draw a curve on the Kendall disk."""
        points_x = [gs.to_numpy(point)[0] for point in self.points]
        points_y = [gs.to_numpy(point)[1] for point in self.points]
        self.ax.plot(points_x, points_y,
                     alpha=alpha, zorder=zorder, **kwargs)

    def draw_vector(self, tangent_vec, base_point, tol=1e-03, **kwargs):
        """Draw one vector in the tangent space to disk at a base point."""
        r_bp, th_bp = self.convert_to_polar_coordinates(base_point)
        bp = gs.array([gs.cos(th_bp) * gs.sin(2 * r_bp),
                       gs.sin(th_bp) * gs.sin(2 * r_bp),
                       gs.cos(2 * r_bp)])
        r_exp, th_exp = self.convert_to_polar_coordinates(
            METRIC_S33.exp(tol * tangent_vec / METRIC_S33.norm(tangent_vec,
                                                               base_point),
                           base_point))
        exp = gs.array([gs.cos(th_exp) * gs.sin(2 * r_exp),
                        gs.sin(th_exp) * gs.sin(2 * r_exp),
                        gs.cos(2 * r_exp)])
        pole = gs.array([0., 0., 1.])

        tv = exp - gs.dot(exp, bp) * bp
        u_tv = tv / gs.linalg.norm(tv)
        u_r = (gs.dot(pole, bp) * bp - pole) / gs.linalg.norm(
            gs.dot(pole, bp) * bp - pole)
        u_th = gs.cross(bp, u_r)
        x_r, x_th = gs.dot(u_tv, u_r), gs.dot(u_tv, u_th)

        bp = self.convert_to_planar_coordinates(base_point)
        u_r = bp / gs.linalg.norm(bp)
        u_th = gs.array([[0., -1.], [1., 0.]]) @ u_r
        tv = METRIC_S33.norm(tangent_vec, base_point) * (x_r * u_r + x_th *
                                                         u_th)

        self.ax.quiver(bp[0], bp[1], tv[0], tv[1], **kwargs)


def convert_to_trihedron(point, space=None):
    """Transform a rigid point into a trihedron.

    Transform a rigid point
    into a trihedron s.t.:
    - the trihedron's base point is the translation of the origin
    of R^3 by the translation part of point,
    - the trihedron's orientation is the rotation of the canonical basis
    of R^3 by the rotation part of point.
    """
    point = gs.to_ndarray(point, to_ndim=2)
    n_points, _ = point.shape

    dim_rotations = SO3_GROUP.dim

    if space == 'SE3_GROUP':
        rot_vec = point[:, :dim_rotations]
        translation = point[:, dim_rotations:]
    elif space == 'SO3_GROUP':
        rot_vec = point
        translation = gs.zeros((n_points, 3))
    else:
        raise NotImplementedError(
            'Trihedrons are only implemented for SO(3) and SE(3).')

    rot_mat = SO3_GROUP.matrix_from_rotation_vector(rot_vec)
    rot_mat = SO3_GROUP.projection(rot_mat)
    basis_vec_1 = gs.array([1., 0., 0.])
    basis_vec_2 = gs.array([0., 1., 0.])
    basis_vec_3 = gs.array([0., 0., 1.])

    trihedrons = []
    for i in range(n_points):
        trihedron_vec_1 = gs.dot(rot_mat[i], basis_vec_1)
        trihedron_vec_2 = gs.dot(rot_mat[i], basis_vec_2)
        trihedron_vec_3 = gs.dot(rot_mat[i], basis_vec_3)
        trihedron = Trihedron(translation[i],
                              trihedron_vec_1,
                              trihedron_vec_2,
                              trihedron_vec_3)
        trihedrons.append(trihedron)
    return trihedrons


def plot(points, ax=None, space=None,
         point_type=None, **point_draw_kwargs):
    """Plot points in one of the implemented manifolds.

    The implemented manifolds are:
    - the special orthogonal group SO(3)
    - the special Euclidean group SE(3)
    - the circle S1 and the sphere S2
    - the hyperbolic plane (the Poincare disk, the Poincare
      half plane and the Klein disk)
    - the Poincare polydisk
    - the Kendall shape space of 2D triangles
    - the Kendall shape space of 3D triangles

    Parameters
    ----------
    points : array-like, shape=[..., dim]
        Points to be plotted.
    space: str, optional, {'SO3_GROUP', 'SE3_GROUP', 'S1', 'S2',
        'H2_poincare_disk', 'H2_poincare_half_plane', 'H2_klein_disk',
        'poincare_polydisk', 'S32', 'M32', 'S33', 'M33'}
    point_type: str, optional, {'extrinsic', 'ball', 'half-space', 'pre-shape'}
    """
    if space not in IMPLEMENTED:
        raise NotImplementedError(
            'The plot function is not implemented'
            ' for space {}. The spaces available for visualization'
            ' are: {}.'.format(space, IMPLEMENTED))

    if points is None:
        raise ValueError("No points given for plotting.")

    if points.ndim < 2:
        points = gs.expand_dims(points, 0)

    if space in ('SO3_GROUP', 'SE3_GROUP'):
        if ax is None:
            ax = plt.subplot(111, projection='3d')
        if space == 'SE3_GROUP':
            ax_s = AX_SCALE * gs.amax(gs.abs(points[:, 3:6]))
        elif space == 'SO3_GROUP':
            ax_s = AX_SCALE * gs.amax(gs.abs(points[:, :3]))
        ax_s = float(ax_s)
        bounds = (-ax_s, ax_s)
        plt.setp(ax,
                 xlim=bounds,
                 ylim=bounds,
                 zlim=bounds,
                 xlabel='X', ylabel='Y', zlabel='Z')
        trihedrons = convert_to_trihedron(points, space=space)
        for t in trihedrons:
            t.draw(ax, **point_draw_kwargs)

    elif space == 'S1':
        circle = Circle()
        ax = circle.set_ax(ax=ax)
        circle.add_points(points)
        circle.draw(ax, **point_draw_kwargs)

    elif space == 'S2':
        sphere = Sphere()
        ax = sphere.set_ax(ax=ax)
        sphere.add_points(points)
        sphere.draw(ax, **point_draw_kwargs)

    elif space == 'H2_poincare_disk':
        if point_type is None:
            point_type = 'extrinsic'
        poincare_disk = PoincareDisk(point_type=point_type)
        ax = poincare_disk.set_ax(ax=ax)
        poincare_disk.add_points(points)
        poincare_disk.draw(ax, **point_draw_kwargs)
        plt.axis('off')

    elif space == 'poincare_polydisk':
        if point_type is None:
            point_type = 'extrinsic'
        n_disks = points.shape[1]
        poincare_poly_disk = PoincarePolyDisk(point_type=point_type,
                                              n_disks=n_disks)
        n_columns = int(gs.ceil(n_disks ** 0.5))
        n_rows = int(gs.ceil(n_disks / n_columns))

        axis_list = []

        for i_disk in range(n_disks):
            axis_list.append(ax.add_subplot(n_rows, n_columns, i_disk + 1))

        for i_disk, one_ax in enumerate(axis_list):
            ax = poincare_poly_disk.set_ax(ax=one_ax)
            poincare_poly_disk.clear_points()
            poincare_poly_disk.add_points(points[:, i_disk, ...])
            poincare_poly_disk.draw(ax, **point_draw_kwargs)

    elif space == 'H2_poincare_half_plane':
        if point_type is None:
            point_type = 'half-space'
        poincare_half_plane = PoincareHalfPlane(point_type=point_type)
        ax = poincare_half_plane.set_ax(points=points, ax=ax)
        poincare_half_plane.add_points(points)
        poincare_half_plane.draw(ax, **point_draw_kwargs)

    elif space == 'H2_klein_disk':
        klein_disk = KleinDisk()
        ax = klein_disk.set_ax(ax=ax)
        klein_disk.add_points(points)
        klein_disk.draw(ax, **point_draw_kwargs)

    elif space == 'SE2_GROUP':
        plane = SpecialEuclidean2()
        ax = plane.set_ax(ax=ax)
        plane.add_points(points)
        plane.draw(ax, **point_draw_kwargs)

    elif space == 'S32':
        sphere = KendallSphere()
        sphere.add_points(points)
        sphere.draw()
        sphere.draw_points()
        ax = sphere.ax

    elif space == 'M32':
        sphere = KendallSphere(point_type='extrinsic')
        sphere.add_points(points)
        sphere.draw()
        sphere.draw_points()
        ax = sphere.ax

    elif space == 'S33':
        disk = KendallDisk()
        disk.add_points(points)
        disk.draw()
        disk.draw_points()
        ax = disk.ax

    elif space == 'M33':
        disk = KendallDisk(point_type='extrinsic')
        disk.add_points(points)
        disk.draw()
        disk.draw_points()
        ax = disk.ax

    return ax
