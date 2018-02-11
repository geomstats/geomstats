"""Visualization for Geometric Statistics."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from geomstats.special_euclidean_group import SpecialEuclideanGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup
import geomstats.special_orthogonal_group as special_orthogonal_group

SE3_GROUP = SpecialEuclideanGroup(n=3)
SO3_GROUP = SpecialOrthogonalGroup(n=3)


class Arrow3D():
    "An arrow in 3d, i.e. a point and a vector."
    def __init__(self, point, vector):
        self.point = point
        self.vector = vector

    def draw(self, ax, **quiver_kwargs):
        "Draw the arrow in 3D plot."
        print('point')
        print(self.point.shape)
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
        self.arrow_1.draw(ax, color='g', **arrow_draw_kwargs)
        self.arrow_2.draw(ax, color='r', **arrow_draw_kwargs)
        self.arrow_3.draw(ax, color='b', **arrow_draw_kwargs)


def trihedron(point, group=SE3_GROUP):
    """
    Transform a rigid pointrmation
    into a trihedron s.t.:
    - the trihedron's base point is the translation of the origin
    of R^3 by the translation part of point,
    - the trihedron's orientation is the rotation of the canonical basis
    of R^3 by the rotation part of point.
    """
    if point.ndim == 1:
        point = np.expand_dims(point, axis=0)
    n_points, _ = point.shape

    dim_rotations = SO3_GROUP.dimension

    if group is SE3_GROUP:
        rot_vec = point[:, :dim_rotations]
        translation = point[:, dim_rotations:]
    elif group is SO3_GROUP:
        rot_vec = point
        translation = np.zeros((n_points, 3))
    else:
        raise NotImplementedError(
                'Visualization is only implemented for SO(3) and SE(3).')

    rot_mat = SO3_GROUP.matrix_from_rotation_vector(rot_vec)
    rot_mat = special_orthogonal_group.closest_rotation_matrix(rot_mat)
    basis_vec_1 = np.array([1, 0, 0])
    basis_vec_2 = np.array([0, 1, 0])
    basis_vec_3 = np.array([0, 0, 1])

    trihedrons = []
    for i in range(n_points):
        trihedron_vec_1 = np.dot(rot_mat[i], basis_vec_1)
        trihedron_vec_2 = np.dot(rot_mat[i], basis_vec_2)
        trihedron_vec_3 = np.dot(rot_mat[i], basis_vec_3)
        trihedron = Trihedron(translation[i],
                              trihedron_vec_1,
                              trihedron_vec_2,
                              trihedron_vec_3)
        trihedrons.append(trihedron)
    return trihedrons


def plot(points, ax=None, group=SE3_GROUP, **point_draw_kwargs):
    """
    Plot points in the 3D Special Euclidean Group,
    by showing them as trihedrons.
    """
    if points is None:
        raise ValueError("No points given for plotting.")

    if points.ndim == 1:
        points = np.expand_dims(points, axis=0)

    if ax is None:
        ax_s = 1.2 * np.amax(np.abs(points[:, 3:6]))
        ax = plt.subplot(111, projection="3d", aspect="equal")
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(-ax_s, ax_s),
                 zlim=(-ax_s, ax_s),
                 xlabel="X", ylabel="Y", zlabel="Z")

    trihedrons = trihedron(points, group=group)
    for t in trihedrons:
        t.draw(ax, **point_draw_kwargs)

    return ax
