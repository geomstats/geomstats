"""
Plot a geodesic on the sphere S2
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere

SPHERE2 = Hypersphere(dimension=2)
METRIC = SPHERE2.metric


def main():
    initial_point = [1., 0., 0.]
    initial_tangent_vec = SPHERE2.projection_to_tangent_space(
        vector=[1., 2., 0.8], base_point=initial_point)
    geodesic = METRIC.geodesic(
        initial_point=initial_point,
        initial_tangent_vec=initial_tangent_vec)

    n_steps = 10
    t = np.linspace(0, 1, n_steps)

    points = geodesic(t)
    visualization.plot(points, space='S2')
    plt.show()


if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] == 'tensorflow':
        logging.info('Examples with visualizations are only implemented '
                     'with numpy backend.\n'
                     'To change backend, write: '
                     'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
