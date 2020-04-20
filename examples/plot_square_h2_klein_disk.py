"""Plot a square on H2 with Klein Disk visualization."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import geomstats.visualization as visualization
from geomstats.geometry.hyperboloid import Hyperboloid

H2 = Hyperboloid(dim=2)
METRIC = H2.metric

SQUARE_SIZE = 50


def main():
    """Plot a square on H2 with Klein Disk visualization."""
    top = SQUARE_SIZE / 2.0
    bot = - SQUARE_SIZE / 2.0
    left = - SQUARE_SIZE / 2.0
    right = SQUARE_SIZE / 2.0
    corners_int = [(bot, left), (bot, right), (top, right), (top, left)]
    corners_ext = H2.from_coordinates(corners_int, 'intrinsic')
    n_steps = 20
    ax = plt.gca()
    for i, src in enumerate(corners_ext):
        dst_id = (i + 1) % len(corners_ext)
        dst = corners_ext[dst_id]
        tangent_vec = METRIC.log(point=dst, base_point=src)
        geodesic = METRIC.geodesic(initial_point=src,
                                   initial_tangent_vec=tangent_vec)
        t = np.linspace(0, 1, n_steps)
        edge_points = geodesic(t)

        visualization.plot(edge_points,
                           ax=ax,
                           space='H2_klein_disk',
                           marker='.',
                           color='black')
    plt.show()


if __name__ == '__main__':
    if os.environ['GEOMSTATS_BACKEND'] == 'tensorflow':
        logging.info('Examples with visualizations are only implemented '
                     'with numpy backend.\n'
                     'To change backend, write: '
                     'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
