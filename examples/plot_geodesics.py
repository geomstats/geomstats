"""
Plot geodesics of the following Riemannian manifolds:
    - SE(3) with its left-invariant canonical metric: a Lie group
"""

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from geomstats.special_euclidean_group import SpecialEuclideanGroup


se3_group = SpecialEuclideanGroup(n=3)
metric = se3_group.left_canonical_metric

initial_point = se3_group.identity
initial_tangent_vec = np.array([1.2, 0.2, 0.3, 6., 0., 0])
geodesic = metric.geodesic(initial_point=initial_point,
                           initial_tangent_vec=initial_tangent_vec)

n_steps = 10
all_t = np.linspace(0, 10, n_steps)

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111, projection='3d', aspect='equal')
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))
ax.set_zlim((-1, 1))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

print(geodesic(all_t))

im = plt.imshow(geodesic(all_t), animated=True)


def updatefig(*args):
    global t
    t += 1
    im.set_array(geodesic(t))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)

plt.show()
