"""
Plot a geodesic of SO(3) equipped
with its left-invariant canonical METRIC.
"""

import matplotlib.pyplot as plt
import numpy as np

from geomstats.special_orthogonal_group import SpecialOrthogonalGroup
import geomstats.visualization as visualization

SO3_GROUP = SpecialOrthogonalGroup(n=3)
METRIC = SO3_GROUP.bi_invariant_metric


def main():
    initial_point = SO3_GROUP.identity
    initial_tangent_vec = [0.5, 0.5, 0.8]
    geodesic = METRIC.geodesic(initial_point=initial_point,
                               initial_tangent_vec=initial_tangent_vec)

    n_steps = 10
    t = np.linspace(0, 1, n_steps)

    points = geodesic(t)
    ax = plt.subplot(111, projection="3d", aspect="equal")
    plt.setp(ax,
             xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
             xlabel="X", ylabel="Y", zlabel="Z")

    visualization.plot(points, ax, space='SO3_GROUP')
    plt.show()


if __name__ == "__main__":
    main()
