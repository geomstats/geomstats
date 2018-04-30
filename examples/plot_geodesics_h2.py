"""
Plot a geodesic on the Hyperbolic space H2,
with Poincare Disk visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

from geomstats.hyperbolic_space import HyperbolicSpace
import geomstats.visualization as visualization

H2 = HyperbolicSpace(dimension=2)
METRIC = H2.metric

def plot_geodesic_between_two_points(initial_point,
                                     end_point, n_steps=10):
    assert H2.belongs(initial_point)
    assert H2.belongs(end_point)

    geodesic = METRIC.geodesic(initial_point=initial_point,
                               end_point=end_point)

    t = np.linspace(0, 1, n_steps)
    points = geodesic(t)
    visualization.plot(points, space='H2')


def plot_geodesic_with_initial_tangent_vector(initial_point,
                                              initial_tangent_vec,
                                              n_steps=10):
    assert H2.belongs(initial_point)
    geodesic = METRIC.geodesic(initial_point=initial_point,
                               initial_tangent_vec=initial_tangent_vec)
    n_steps = 10
    t = np.linspace(0, 1, n_steps)

    points = geodesic(t)
    visualization.plot(points, space='H2')


def main():
    initial_point = [np.sqrt(2), 1., 0.]
    end_point = H2.intrinsic_to_extrinsic_coords([1.5, 1.5])
    initial_tangent_vec = H2.projection_to_tangent_space(
                                        vector=[3.5, 0.6, 0.8],
                                        base_point=initial_point)

    plot_geodesic_between_two_points(initial_point, end_point)
    plot_geodesic_with_initial_tangent_vector(initial_point, initial_tangent_vec)
    plt.show()

if __name__ == "__main__":
    main()
