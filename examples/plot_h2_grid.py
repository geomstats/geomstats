"""
Plot a grid on H2
with Poincare Disk visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

from geomstats.hyperbolic_space import HyperbolicSpace
import geomstats.visualization as visualization

H2 = HyperbolicSpace(dimension=2)
METRIC = H2.metric

c = H2.intrinsic_to_extrinsic_coords


def main():
    starts = []
    ends = []
    left = -128
    right = 128
    bottom = -128
    top = 128
    grid_size = 32
    n_steps = 512
    for p in np.linspace(left, right, grid_size):
        starts.append(np.array([top, p]))
        ends.append(np.array([bottom, p]))
    for p in np.linspace(top, bottom, grid_size):
        starts.append(np.array([p, left]))
        ends.append(np.array([p, right]))
    starts = [c(s) for s in starts]
    ends = [c(e) for e in ends]
    for i, _ in enumerate(starts):
        initial_tangent_vec = METRIC.log(ends[i], starts[i])
        geodesic = METRIC.geodesic(initial_point=starts[i],
                                   initial_tangent_vec=initial_tangent_vec)

        t = np.linspace(0, 1, n_steps)
        points_to_plot = geodesic(t)
        visualization.plot(points_to_plot, space='H2', marker='.', s=1)
    plt.show()


if __name__ == "__main__":
    main()
