"""
Plot geodesics of the following Riemannian manifolds:
    - SE(3) with its left-invariant canonical metric: a Lie group
"""

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from geomstats.special_euclidean_group import SpecialEuclideanGroup
import geomstats.visualization as visualization


def plot_point(ax, point, space):
    trihedron = visualization.trihedron_from_rigid_transformation(point)
    trihedron.draw(ax)
    return ax


def plot_geodesic(ax, points, space=None):
    if points is None or len(points) == 0:
        raise ValueError("Geodesic does not contain any elements.")

    if ax is None:
        ax_s = np.amax(np.abs(points[:, 3:6]))
        ax = plt.subplot(111, projection="3d", aspect="equal")
        plt.setp(ax,
                 xlim=(-ax_s, ax_s),
                 ylim=(-ax_s, ax_s),
                 zlim=(-ax_s, ax_s),
                 xlabel="X", ylabel="Y", zlabel="Z")

    for point in points:
        plot_point(ax=ax, point=point, space=space)

    return ax

se3_group = SpecialEuclideanGroup(n=3)
metric = se3_group.left_canonical_metric

initial_point = se3_group.identity
initial_tangent_vec = np.array([1.2, 0.2, 0.3, 6., 0., 0])
geodesic = metric.geodesic(initial_point=initial_point,
                           initial_tangent_vec=initial_tangent_vec)

n_steps = 10
all_t = np.linspace(0, 10, n_steps)
points = np.vstack([geodesic(t) for t in all_t])

fig = plt.figure(figsize=(15, 5))

ax = fig.add_subplot(111, projection="3d", aspect="equal")
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))
ax.set_zlim((-1, 1))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

trihedron_start = visualization.trihedron_from_rigid_transformation(points[0])
trihedron_end = visualization.trihedron_from_rigid_transformation(points[-1])

im = plt.imshow(points[all_t, :], animated=True)


def updatefig(*args):
    global t
    t += 1
    im.set_array(points[t, :])
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)

plt.show()
