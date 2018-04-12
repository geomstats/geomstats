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
    initial_point = [np.sqrt(2), 1., 0.]
    assert H2.belongs(initial_point)
    initial_tangent_vec = H2.projection_to_tangent_space(
                                        vector=[3.5, 0.6, 0.8],
                                        base_point=initial_point)
    print('initial_tangent_vec: {}'.format(initial_tangent_vec))
    geodesic = METRIC.geodesic(initial_point=initial_point,
                               initial_tangent_vec=initial_tangent_vec)

    initial_tangent_step_1 = H2.extrinsic_to_intrinsic_coords(initial_tangent_vec) / 10
    print('initial_tangent_vec / 10 : {}'.format(initial_tangent_step_1))
    step_1 = METRIC.exp(tangent_vec=initial_tangent_step_1, base_point=initial_point)
    print('step 1: {}'.format(step_1))

    initial_tangent_step_2 = 2* initial_tangent_step_1
    step_2 = METRIC.exp(tangent_vec=initial_tangent_step_2, base_point=initial_point)
    print('step 2: {}'.format(step_2))

    n_steps = 10
    t = np.linspace(0, 1, n_steps)

    points = geodesic(t)

    visualization.plot(points, space='H2')
    plt.show()


if __name__ == "__main__":
    main()
