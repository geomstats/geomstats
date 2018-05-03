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


def main(left=-128,
         right=128,
         bottom=-128,
         top=128,
         grid_size=32,
         n_steps=512):
    starts = []
    ends = []
    for p in np.linspace(left, right, grid_size):
        starts.append(np.array([top, p]))
        ends.append(np.array([bottom, p]))
    for p in np.linspace(top, bottom, grid_size):
        starts.append(np.array([p, left]))
        ends.append(np.array([p, right]))
    starts = [H2.intrinsic_to_extrinsic_coords(s) for s in starts]
    ends = [H2.intrinsic_to_extrinsic_coords(e) for e in ends]
    for start, end in zip(starts, ends):
        geodesic = METRIC.geodesic(initial_point=start,
                                   end_point=end)

        t = np.linspace(0, 1, n_steps)
        points_to_plot = geodesic(t)
        visualization.plot(points_to_plot, space='H2', marker='.', s=1)
    plt.show()


if __name__ == "__main__":
    main()
