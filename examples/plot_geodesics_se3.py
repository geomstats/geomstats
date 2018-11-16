"""
Plot a geodesic of SE(3) equipped
with its left-invariant canonical METRIC.
"""

import matplotlib.pyplot as plt
import numpy as np

import geomstats.visualization as visualization

from geomstats.special_euclidean_group import SpecialEuclideanGroup

SE3_GROUP = SpecialEuclideanGroup(n=3)
METRIC = SE3_GROUP.left_canonical_metric


def main():
    initial_point = SE3_GROUP.identity
    initial_tangent_vec = [1.8, 0.2, 0.3, 3., 3., 1.]
    geodesic = METRIC.geodesic(initial_point=initial_point,
                               initial_tangent_vec=initial_tangent_vec)

    n_steps = 40
    t = np.linspace(0, 5, n_steps)

    points = geodesic(t)
    ax = plt.subplot(111, projection="3d", aspect="equal")
    plt.setp(ax, xlim=(-1, 18), ylim=(-1, 18), zlim=(-1, 5),
             xlabel='X', ylabel='Y', zlabel='Z')
    ax.tick_params(axis='both', labelsize=7)

    visualization.plot(points, ax, space='SE3_GROUP')
    plt.show()


if __name__ == "__main__":
    main()
