"""The Visualization Package."""

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import geomstats.backend as gs
from geomstats.visualization.hyperbolic import (
    KleinDisk,
    PoincareDisk,
    PoincareHalfPlane,
)
from geomstats.visualization.hypersphere import Circle, Sphere
from geomstats.visualization.poincare_polydisk import PoincarePolyDisk
from geomstats.visualization.pre_shape import KendallDisk, KendallSphere
from geomstats.visualization.spd_matrices import Ellipses
from geomstats.visualization.special_euclidean import SpecialEuclidean2
from geomstats.visualization.special_orthogonal import (
    Arrow3D,
    Trihedron,
    convert_to_trihedron,
)

AX_SCALE = 1.2


IMPLEMENTED = [
    "SO3_GROUP",
    "SE3_GROUP",
    "SE2_GROUP",
    "S1",
    "S2",
    "H2_poincare_disk",
    "H2_poincare_half_plane",
    "H2_klein_disk",
    "poincare_polydisk",
    "S32",
    "M32",
    "S33",
    "M33",
    "SPD2",
]


def tutorial_matplotlib():
    """Configure style for matplotlib tutorial."""
    fontsize = 12
    matplotlib.rc("font", size=fontsize)
    matplotlib.rc("text")
    matplotlib.rc("legend", fontsize=fontsize)
    matplotlib.rc("axes", titlesize=21, labelsize=14)
    matplotlib.rc(
        "font",
        family="times",
        serif=["Computer Modern Roman"],
        monospace=["Computer Modern Typewriter"],
    )


def plot(points, ax=None, space=None, point_type=None, **point_draw_kwargs):
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
        'poincare_polydisk', 'S32', 'M32', 'S33', 'M33', 'SPD2'}
    point_type: str, optional, {'extrinsic', 'ball', 'half-space', 'pre-shape'}
    """
    if space not in IMPLEMENTED:
        raise NotImplementedError(
            "The plot function is not implemented"
            " for space {}. The spaces available for visualization"
            " are: {}.".format(space, IMPLEMENTED)
        )

    if points is None:
        raise ValueError("No points given for plotting.")

    if points.ndim < 2:
        points = gs.expand_dims(points, 0)

    if space in ("SO3_GROUP", "SE3_GROUP"):
        if ax is None:
            ax = plt.subplot(111, projection="3d")
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

    elif space == "S1":
        circle = Circle()
        ax = circle.set_ax(ax=ax)
        circle.add_points(points)
        circle.draw(ax, **point_draw_kwargs)

    elif space == "S2":
        sphere = Sphere()
        ax = sphere.set_ax(ax=ax)
        sphere.add_points(points)
        sphere.draw(ax, **point_draw_kwargs)

    elif space == "H2_poincare_disk":
        if point_type is None:
            point_type = "extrinsic"
        poincare_disk = PoincareDisk(point_type=point_type)
        ax = poincare_disk.set_ax(ax=ax)
        poincare_disk.add_points(points)
        poincare_disk.draw(ax, **point_draw_kwargs)
        plt.axis("off")

    elif space == "poincare_polydisk":
        if point_type is None:
            point_type = "extrinsic"
        n_disks = points.shape[1]
        poincare_poly_disk = PoincarePolyDisk(point_type=point_type, n_disks=n_disks)
        n_columns = int(gs.ceil(n_disks**0.5))
        n_rows = int(gs.ceil(n_disks / n_columns))

        axis_list = []

        for i_disk in range(n_disks):
            axis_list.append(ax.add_subplot(n_rows, n_columns, i_disk + 1))

        for i_disk, one_ax in enumerate(axis_list):
            ax = poincare_poly_disk.set_ax(ax=one_ax)
            poincare_poly_disk.clear_points()
            poincare_poly_disk.add_points(points[:, i_disk, ...])
            poincare_poly_disk.draw(ax, **point_draw_kwargs)

    elif space == "H2_poincare_half_plane":
        if point_type is None:
            point_type = "half-space"
        poincare_half_plane = PoincareHalfPlane(point_type=point_type)
        ax = poincare_half_plane.set_ax(ax=ax)
        poincare_half_plane.add_points(points)
        poincare_half_plane.draw(ax, **point_draw_kwargs)

    elif space == "H2_klein_disk":
        klein_disk = KleinDisk()
        ax = klein_disk.set_ax(ax=ax)
        klein_disk.add_points(points)
        klein_disk.draw(ax, **point_draw_kwargs)

    elif space == "SE2_GROUP":
        plane = SpecialEuclidean2()
        ax = plane.set_ax(ax=ax)
        plane.add_points(points)
        plane.draw_points(ax, **point_draw_kwargs)

    elif space == "S32":
        sphere = KendallSphere()
        sphere.add_points(points)
        sphere.draw()
        sphere.draw_points()
        ax = sphere.ax

    elif space == "M32":
        sphere = KendallSphere(point_type="extrinsic")
        sphere.add_points(points)
        sphere.draw()
        sphere.draw_points()
        ax = sphere.ax

    elif space == "S33":
        disk = KendallDisk()
        disk.add_points(points)
        disk.draw()
        disk.draw_points()
        ax = disk.ax

    elif space == "M33":
        disk = KendallDisk(point_type="extrinsic")
        disk.add_points(points)
        disk.draw()
        disk.draw_points()
        ax = disk.ax

    elif space == "SPD2":
        ellipses = Ellipses()
        ellipses.draw_points(points=points)

    return ax
