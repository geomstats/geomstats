"""
Plot a geodesic of SE(3) equipped
with its left-invariant canonical METRIC.
"""

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from geomstats.special_euclidean_group import SpecialEuclideanGroup
import geomstats.visualization as visualization

SE3_GROUP = SpecialEuclideanGroup(n=3)
METRIC = SE3_GROUP.left_canonical_metric


def main():
    initial_point = SE3_GROUP.identity
    initial_tangent_vec = np.array([1.8, 0.2, 0.3, 3., 3., 1.])
    geodesic = METRIC.geodesic(initial_point=initial_point,
                               initial_tangent_vec=initial_tangent_vec)

    n_steps = 10
    t = np.linspace(0, 10, n_steps)
    points = geodesic(t)

    fig = plt.figure(figsize=(15, 5))

    im = plt.imshow(geodesic(t), animated=True)

    def updatefig(*args):
        global t
        t += 1
        im.set_array(geodesic(t))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
