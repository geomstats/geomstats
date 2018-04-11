"""
Plot a square on a h2 space,
with Poincare Disk visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

from geomstats.hyperbolic_space import HyperbolicSpace
import geomstats.visualization as visualization

H2 = HyperbolicSpace(dimension=2)
METRIC = H2.metric


def main():
    points = []
    points.append(H2.intrinsic_to_extrinsic_coords(np.array([-2.999, -2.999])))
    points.append(H2.intrinsic_to_extrinsic_coords(np.array([-2.999, 2.999])))
    points.append(H2.intrinsic_to_extrinsic_coords(np.array([2.999, 2.999])))
    points.append(H2.intrinsic_to_extrinsic_coords(np.array([2.999, -2.999])))
    for i in range(0, 4):
        dst = (i+1) % 4
        initial_tangent_vec = METRIC.log(points[dst], points[i])
        geodesic = METRIC.geodesic(initial_point=points[i],
                                   initial_tangent_vec=initial_tangent_vec)

        n_steps = 10
        t = np.linspace(0, 1, n_steps)
        points_to_plot = geodesic(t)
        visualization.plot(points_to_plot, space='H2', line='-')
    plt.show()


if __name__ == "__main__":
    main()
