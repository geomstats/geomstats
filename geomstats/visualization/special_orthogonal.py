"""Visualization for Geometric Statistics."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

SO3_GROUP = SpecialOrthogonal(n=3, point_type="vector")


AX_SCALE = 1.2


class Arrow3D:
    """An arrow in 3d, i.e. a point and a vector."""

    def __init__(self, point, vector):
        self.point = point
        self.vector = vector

    def draw(self, ax, **quiver_kwargs):
        """Draw the arrow in 3D plot."""
        ax.quiver(
            self.point[0],
            self.point[1],
            self.point[2],
            self.vector[0],
            self.vector[1],
            self.vector[2],
            **quiver_kwargs
        )


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
        if "color" in arrow_draw_kwargs:
            self.arrow_1.draw(ax, **arrow_draw_kwargs)
            self.arrow_2.draw(ax, **arrow_draw_kwargs)
            self.arrow_3.draw(ax, **arrow_draw_kwargs)
        else:
            blue = "#1f77b4"
            orange = "#ff7f0e"
            green = "#2ca02c"
            self.arrow_1.draw(ax, color=blue, **arrow_draw_kwargs)
            self.arrow_2.draw(ax, color=orange, **arrow_draw_kwargs)
            self.arrow_3.draw(ax, color=green, **arrow_draw_kwargs)

    def plot(self, points, ax=None, space=None, **point_draw_kwargs):
        if space == "SE3_GROUP":
            ax_s = AX_SCALE * gs.amax(gs.abs(points[:, 3:6]))
        elif space == "SO3_GROUP":
            ax_s = AX_SCALE * gs.amax(gs.abs(points[:, :3]))
        ax_s = float(ax_s)
        bounds = (-ax_s, ax_s)
        plt.setp(
            ax,
            xlim=bounds,
            ylim=bounds,
            zlim=bounds,
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )
        trihedrons = convert_to_trihedron(points, space=space)
        for t in trihedrons:
            t.draw(ax, **point_draw_kwargs)


def convert_to_trihedron(point, space=None):
    """Transform a rigid point into a trihedron.

    Transform a rigid point into a trihedron such that:
    - the trihedron's base point is the translation of the origin
    of R^3 by the translation part of point,
    - the trihedron's orientation is the rotation of the canonical basis
    of R^3 by the rotation part of point.
    """
    point = gs.to_ndarray(point, to_ndim=2)
    n_points, _ = point.shape

    dim_rotations = SO3_GROUP.dim

    if space == "SE3_GROUP":
        rot_vec = point[:, :dim_rotations]
        translation = point[:, dim_rotations:]
    elif space == "SO3_GROUP":
        rot_vec = point
        translation = gs.zeros((n_points, 3))
    else:
        raise NotImplementedError(
            "Trihedrons are only implemented for SO(3) and SE(3)."
        )

    rot_mat = SO3_GROUP.matrix_from_rotation_vector(rot_vec)
    rot_mat = SO3_GROUP.projection(rot_mat)
    basis_vec_1 = gs.array([1.0, 0.0, 0.0])
    basis_vec_2 = gs.array([0.0, 1.0, 0.0])
    basis_vec_3 = gs.array([0.0, 0.0, 1.0])

    trihedrons = []
    for i in range(n_points):
        trihedron_vec_1 = gs.dot(rot_mat[i], basis_vec_1)
        trihedron_vec_2 = gs.dot(rot_mat[i], basis_vec_2)
        trihedron_vec_3 = gs.dot(rot_mat[i], basis_vec_3)
        trihedron = Trihedron(
            translation[i], trihedron_vec_1, trihedron_vec_2, trihedron_vec_3
        )
        trihedrons.append(trihedron)
    return trihedrons


def plot(points, ax=None, space=None, **point_draw_kwargs):
    """Plot trihedrons."""
    ax_s = AX_SCALE * gs.amax(gs.abs(points[:, :3]))
    ax_s = float(ax_s)
    bounds = (-ax_s, ax_s)
    plt.setp(
        ax,
        xlim=bounds,
        ylim=bounds,
        zlim=bounds,
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
    )
    trihedrons = convert_to_trihedron(points, space=space)
    for t in trihedrons:
        t.draw(ax, **point_draw_kwargs)
