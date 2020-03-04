"""
Plot a geodesic of SE(3) equipped
with its left-invariant canonical METRIC.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean

SE3_GROUP = SpecialEuclidean(n=3)
METRIC = SE3_GROUP.left_canonical_metric


def main():
    initial_point = SE3_GROUP.identity
    initial_tangent_vec = [1.8, 0.2, 0.3, 3., 3., 1.]
    geodesic = METRIC.geodesic(initial_point=initial_point,
                               initial_tangent_vec=initial_tangent_vec)

    n_steps = 40
    t = np.linspace(-3, 3, n_steps)

    points = geodesic(t)

    visualization.plot(points, space='SE3_GROUP')
    plt.show()


if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] == 'tensorflow':
        logging.info('Examples with visualizations are only implemented '
                     'with numpy backend.\n'
                     'To change backend, write: '
                     'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
