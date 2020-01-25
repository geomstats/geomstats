"""Visualization for Geometric Statistics."""

import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from mpl_toolkits.mplot3d import Axes3D  # NOQA

SE3_GROUP = SpecialEuclidean(n=3)
SO3_GROUP = SpecialOrthogonal(n=3)
S1 = Hypersphere(dimension=1)
S2 = Hypersphere(dimension=2)
H2 = Hyperbolic(dimension=2)

AX_SCALE = 1.2

IMPLEMENTED = ['SO3_GROUP', 'SE3_GROUP', 'S1', 'S2',
               'H2_poincare_disk', 'H2_poincare_half_plane', 'H2_klein_disk']


class Arrow3D():
    "An arrow in 3d, i.e. a point and a vector."
    def __init__(self, point, vector):
        self.point = point
        self.vector = vector

    def draw(self, ax, **quiver_kwargs):
        "Draw the arrow in 3D plot."
        ax.quiver(self.point[0], self.point[1], self.point[2],
                  self.vector[0], self.vector[1], self.vector[2],
                  **quiver_kwargs)


class Trihedron():
    "A trihedron, i.e. 3 Arrow3Ds at the same point."
    def __init__(self, point, vec_1, vec_2, vec_3):
        self.arrow_1 = Arrow3D(point, vec_1)
        self.arrow_2 = Arrow3D(point, vec_2)
        self.arrow_3 = Arrow3D(point, vec_3)

    def draw(self, ax, **arrow_draw_kwargs):
        """
        Draw the trihedron by drawing its 3 Arrow3Ds.
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


class Circle():
    def __init__(self, n_angles=100, points=None):
        angles = gs.linspace(0, 2 * gs.pi, n_angles)
        self.circle_x = gs.cos(angles)
        self.circle_y = gs.sin(angles)
        self.points = []
        if points is not None:
            self.add_points(points)

    def set_ax(self, ax=None):
        if ax is None:
            ax = plt.subplot()
        ax_s = AX_SCALE
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(-ax_s, ax_s),
                 xlabel='X', ylabel='Y')
        return ax

    def add_points(self, points):
        assert gs.all(S1.belongs(points))
        if not isinstance(points, list):
            points = points.tolist()
        self.points.extend(points)

    def draw(self, ax, **plot_kwargs):
        ax.plot(self.circle_x, self.circle_y, color="black")
        if self.points:
            self.draw_points(ax, **plot_kwargs)

    def draw_points(self, ax, points=None, **plot_kwargs):
        if points is None:
            points = self.points
        else:
            points = points
        points = gs.array(points)
        ax.plot(points[:, 0], points[:, 1], marker='o', linestyle="None",
                **plot_kwargs)


class Sphere():
    """
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

    def set_ax(self, ax=None):
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
        assert gs.all(S2.belongs(points))
        if not isinstance(points, list):
            points = points.tolist()
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
        else:
            points = points
        points_x = gs.vstack([point[0] for point in points])
        points_y = gs.vstack([point[1] for point in points])
        points_z = gs.vstack([point[2] for point in points])
        ax.scatter(points_x, points_y, points_z, **scatter_kwargs)

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


class PoincareDisk():
    def __init__(self, points=None, point_type='extrinsic'):
        self.center = gs.array([0., 0.])
        self.points = []
        self.point_type = point_type

        if points is not None:
            self.add_points(points)

    def set_ax(self, ax=None):
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
            assert gs.all(H2.belongs(points))
            points = self.convert_to_poincare_coordinates(points)

        if not isinstance(points, list):
            points = points.tolist()
        self.points.extend(points)

    def convert_to_poincare_coordinates(self, points):
        poincare_coords = points[:, 1:] / (1 + points[:, :1])
        return poincare_coords

    def draw(self, ax, **kwargs):
        circle = plt.Circle((0, 0), radius=1., color='black', fill=False)
        ax.add_artist(circle)
        points_x = gs.vstack([point[0] for point in self.points])
        points_y = gs.vstack([point[1] for point in self.points])
        ax.scatter(points_x, points_y, **kwargs)


class PoincareHalfPlane():
    def __init__(self, points=None):
        self.points = []
        if points is not None:
            self.add_points(points)

    def add_points(self, points):
        assert gs.all(H2.belongs(points))
        points = self.convert_to_half_plane_coordinates(points)
        if not isinstance(points, list):
            points = points.tolist()
        self.points.extend(points)

    def set_ax(self, ax=None):
        if ax is None:
            ax = plt.subplot()
        ax_s = AX_SCALE
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(0., ax_s),
                 xlabel='X', ylabel='Y')
        return ax

    def convert_to_half_plane_coordinates(self, points):
        disk_coords = points[:, 1:] / (1 + points[:, :1])
        disk_x = disk_coords[:, 0]
        disk_y = disk_coords[:, 1]

        half_plane_coords = gs.zeros_like(disk_coords)
        denominator = (disk_x ** 2 + (1 - disk_y) ** 2)
        half_plane_coords[:, 0] = 2 * disk_x / denominator
        half_plane_coords[:, 1] = ((1 - disk_x ** 2 - disk_y ** 2)
                                   / denominator)
        return half_plane_coords

    def draw(self, ax, **kwargs):
        points_x = gs.vstack([point[0] for point in self.points])
        points_y = gs.vstack([point[1] for point in self.points])
        ax.scatter(points_x, points_y, **kwargs)


class KleinDisk():
    def __init__(self, points=None):
        self.center = gs.array([0., 0.])
        self.points = []
        if points is not None:
            self.add_points(points)

    def set_ax(self, ax=None):
        if ax is None:
            ax = plt.subplot()
        ax_s = AX_SCALE
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(-ax_s, ax_s),
                 xlabel='X', ylabel='Y')
        return ax

    def add_points(self, points):
        assert gs.all(H2.belongs(points))
        points = self.convert_to_klein_coordinates(points)
        if not isinstance(points, list):
            points = points.tolist()
        self.points.extend(points)

    def convert_to_klein_coordinates(self, points):
        poincare_coords = points[:, 1:] / (1 + points[:, :1])
        poincare_radius = gs.linalg.norm(
            poincare_coords, axis=1)
        poincare_angle = gs.arctan2(
            poincare_coords[:, 1], poincare_coords[:, 0])

        klein_radius = 2 * poincare_radius / (1 + poincare_radius ** 2)
        klein_angle = poincare_angle

        klein_coords = gs.zeros_like(poincare_coords)
        klein_coords[:, 0] = klein_radius * gs.cos(klein_angle)
        klein_coords[:, 1] = klein_radius * gs.sin(klein_angle)
        return klein_coords

    def draw(self, ax, **kwargs):
        circle = plt.Circle((0, 0), radius=1., color='black', fill=False)
        ax.add_artist(circle)
        points_x = gs.vstack([point[0] for point in self.points])
        points_y = gs.vstack([point[1] for point in self.points])
        ax.scatter(points_x, points_y, **kwargs)


def convert_to_trihedron(point, space=None):
    """
    Transform a rigid pointrmation
    into a trihedron s.t.:
    - the trihedron's base point is the translation of the origin
    of R^3 by the translation part of point,
    - the trihedron's orientation is the rotation of the canonical basis
    of R^3 by the rotation part of point.
    """
    point = gs.to_ndarray(point, to_ndim=2)
    n_points, _ = point.shape

    dim_rotations = SO3_GROUP.dimension

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
    basis_vec_1 = gs.array([1, 0, 0])
    basis_vec_2 = gs.array([0, 1, 0])
    basis_vec_3 = gs.array([0, 0, 1])

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
    """
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

    points = gs.to_ndarray(points, to_ndim=2)

    if space in ('SO3_GROUP', 'SE3_GROUP'):
        if ax is None:
            ax = plt.subplot(111, projection='3d')
        if space == 'SE3_GROUP':
            ax_s = AX_SCALE * gs.amax(gs.abs(points[:, 3:6]))
        elif space == 'SO3_GROUP':
            ax_s = AX_SCALE * gs.amax(gs.abs(points[:, :3]))
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(-ax_s, ax_s),
                 zlim=(-ax_s, ax_s),
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
