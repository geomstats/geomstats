"""
Plot a geodesic on the sphere S2
"""

import numpy as np
import matplotlib.pyplot as plt

from geomstats.hypersphere import Hypersphere
import geomstats.visualization as visualization

SPHERE2 = Hypersphere(dimension=2)
METRIC = SPHERE2.metric


def main():
    initial_point = [1., 0., 0.]
    initial_tangent_vec = SPHERE2.projection_to_tangent_space(
                                        vector=[1., 2., 0.8],
                                        base_point=initial_point)
    geodesic = METRIC.geodesic(initial_point=initial_point,
                               initial_tangent_vec=initial_tangent_vec)

    n_steps = 10
    t = np.linspace(0, 1, n_steps)

    points = geodesic(t)
    ax = plt.subplot(111, projection="3d", aspect="equal")

    visualization.plot(points, ax, space='S2')
    plt.show()


if __name__ == "__main__":
    main()
