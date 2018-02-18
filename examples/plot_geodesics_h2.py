"""
Plot a geodesic on the sphere S2
"""

import numpy as np
import matplotlib.pyplot as plt

from geomstats.hyperbolic_space import HyperbolicSpace
import geomstats.visualization as visualization

H2 = HyperbolicSpace(dimension=2)
METRIC = H2.metric


def main():
    initial_point = np.array([np.sqrt(10), 3., 0.])
    assert H2.belongs(initial_point)
    initial_tangent_vec = H2.projection_to_tangent_space(
                                        vector=np.array([1., 2., 0.8]),
                                        base_point=initial_point)
    geodesic = METRIC.geodesic(initial_point=initial_point,
                               initial_tangent_vec=initial_tangent_vec)

    n_steps = 100
    t = np.linspace(0, 1, n_steps)

    points = geodesic(t)
    bool_belongs = H2.belongs(points)
    print(bool_belongs)

    visualization.plot(points, space='H2')
    plt.show()


if __name__ == "__main__":
    main()
