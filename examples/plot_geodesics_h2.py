"""
Plot a geodesic on the hyperbolic space h2,
with Poincare Disk visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

from geomstats.hyperbolic_space import HyperbolicSpace
import geomstats.visualization as visualization

H2 = HyperbolicSpace(dimension=2)
METRIC = H2.metric


def main():
    initial_point = [1., 0., 0.]
    assert H2.belongs(initial_point)
    initial_tangent_vec = H2.projection_to_tangent_space(
                                        vector=[0., 300., 0.],
                                        base_point=initial_point)
    geodesic = METRIC.geodesic(initial_point=initial_point,
                               initial_tangent_vec=initial_tangent_vec)

    n_steps = 10
    t = np.linspace(0, 1, n_steps)

    points = geodesic(t)

    visualization.plot(points, space='H2')
    plt.show()


if __name__ == "__main__":
    main()
