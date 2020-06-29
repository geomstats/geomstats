"""Visualization for Geometric Statistics."""

import matplotlib
import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from mpl_toolkits.mplot3d import Axes3D  # NOQA

SE3_GROUP = SpecialEuclidean(n=3, point_type='vector')
SO3_GROUP = SpecialOrthogonal(n=3, point_type='vector')
S1 = Hypersphere(dim=1)
S2 = Hypersphere(dim=2)
H2 = Hyperboloid(dim=2)

AX_SCALE = 1.2

IMPLEMENTED = ['SO3_GROUP', 'SE3_GROUP', 'S1', 'S2',
               'H2_poincare_disk', 'H2_poincare_half_plane', 'H2_klein_disk',
               'poincare_polydisk']


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

    def __init__(self, points=None):
        self.points = []
        if points is not None:
            self.add_points(points)

    def add_points(self, points):
        if not gs.all(H2.belongs(points)):
            raise ValueError(
                'Points do not belong to the hyperbolic space.')
        points = self.convert_to_half_plane_coordinates(points)
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    @staticmethod
    def set_ax(ax=None):
        if ax is None:
            ax = plt.subplot()
        ax_s = AX_SCALE
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(0., ax_s),
                 xlabel='X', ylabel='Y')
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
         point_type='extrinsic', **point_draw_kwargs):
    """Plot points in the 3D Special Euclidean Group.

    Plot points in the 3D Special Euclidean Group,
    by showing them as trihedrons.
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
        poincare_disk = PoincareDisk(point_type=point_type)
        ax = poincare_disk.set_ax(ax=ax)
        poincare_disk.add_points(points)
        poincare_disk.draw(ax, **point_draw_kwargs)
        plt.axis('off')

    elif space == 'poincare_polydisk':
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
        poincare_half_plane = PoincareHalfPlane()
        ax = poincare_half_plane.set_ax(ax=ax)
        poincare_half_plane.add_points(points)
        poincare_half_plane.draw(ax, **point_draw_kwargs)

    elif space == 'H2_klein_disk':
        klein_disk = KleinDisk()
        ax = klein_disk.set_ax(ax=ax)
        klein_disk.add_points(points)
        klein_disk.draw(ax, **point_draw_kwargs)

    return ax
