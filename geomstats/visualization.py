"""Visualization for Geometric Statistics."""

import matplotlib
matplotlib.use('PDF')  # noqa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import geomstats.special_orthogonal_group as special_orthogonal_group


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
        self.arrow_1.draw(ax, color='g', **arrow_draw_kwargs)
        self.arrow_2.draw(ax, color='r', **arrow_draw_kwargs)
        self.arrow_3.draw(ax, color='b', **arrow_draw_kwargs)


def trihedron_from_rigid_transformation(transfo):
    """
    Transform a rigid transformation
    into a trihedron s.t.:
    - the trihedron's base point is the translation of the origin
    of R^3 by the translation part of transfo,
    - the trihedron's orientation is the rotation of the canonical basis
    of R^3 by the rotation part of transfo.
    """
    SO3 = special_orthogonal_group.SpecialOrthogonalGroup(3)
    translation = transfo[3:6]
    rot_vec = transfo[0:3]
    rot_mat = SO3.matrix_from_rotation_vector(rot_vec)
    rot_mat = special_orthogonal_group.closest_rotation_matrix(rot_mat)

    basis_vec_1 = np.array([1, 0, 0])
    basis_vec_2 = np.array([0, 1, 0])
    basis_vec_3 = np.array([0, 0, 1])

    trihedron_vec_1 = np.dot(rot_mat, basis_vec_1)
    trihedron_vec_2 = np.dot(rot_mat, basis_vec_2)
    trihedron_vec_3 = np.dot(rot_mat, basis_vec_3)
    trihedron = Trihedron(translation,
                          trihedron_vec_1,
                          trihedron_vec_2,
                          trihedron_vec_3)

    return trihedron


def plot_trihedron(trihedron, **trihedron_draw_kwargs):
    "Return the figure with the plotted trihedron."
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    trihedron.draw(ax, **trihedron_draw_kwargs)
    return fig
