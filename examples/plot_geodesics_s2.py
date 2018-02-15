"""
Plot a geodesic on the sphere S2
"""

import numpy as np
import matplotlib.pyplot as plt

from geomstats.hypersphere import Hypersphere
import geomstats.visualization as visualization

SPHERE2 = Hypersphere(dimension=2)


def main():
    initial_point = np.array([1., 0., 0.])
    initial_tangent_vec = np.array([1.8, 0.2, 0.3, 3., 3., 1.])
    geodesic = METRIC.geodesic(initial_point=initial_point,
                               initial_tangent_vec=initial_tangent_vec)

    n_steps = 10
    t = np.linspace(0, 1, n_steps)

    points = geodesic(t)
    ax = plt.subplot(111, projection="3d", aspect="equal")
    plt.setp(ax,
             xlim=(-1, 4), ylim=(-1, 4), zlim=(-1, 2),
             xlabel="X", ylabel="Y", zlabel="Z")

    visualization.plot(points, ax, space='SE3_GROUP')
    plt.show()


if __name__ == "__main__":
    main()
