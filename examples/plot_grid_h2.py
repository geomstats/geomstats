"""Plot a grid on H2 with Poincare Disk visualization."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import geomstats.visualization as visualization
from geomstats.geometry.hyperboloid import Hyperboloid

H2 = Hyperboloid(dim=2)
METRIC = H2.metric


def main(left=-128,
         right=128,
         bottom=-128,
         top=128,
         grid_size=32,
         n_steps=512):
    """Plot a grid on H2 with Poincare Disk visualization.

    Parameters
    ----------
    left, right, bottom, top : ints
        Grid's coordinates
    grid_size : int
        Grid's size.
    n_steps : int
        Number of steps along the geodesics defining the grid.
    """
    starts = []
    ends = []
    for p in np.linspace(left, right, grid_size):
        starts.append(np.array([top, p]))
        ends.append(np.array([bottom, p]))
    for p in np.linspace(top, bottom, grid_size):
        starts.append(np.array([p, left]))
        ends.append(np.array([p, right]))
    starts = [H2.from_coordinates(s, 'intrinsic') for s in starts]
    ends = [H2.from_coordinates(e, 'intrinsic') for e in ends]
    ax = plt.gca()
    for start, end in zip(starts, ends):
        geodesic = METRIC.geodesic(initial_point=start,
                                   end_point=end)

        t = np.linspace(0, 1, n_steps)
        points_to_plot = geodesic(t)
        visualization.plot(
            points_to_plot, ax=ax, space='H2_poincare_disk', marker='.', s=1)
    plt.show()


if __name__ == '__main__':
    if os.environ['GEOMSTATS_BACKEND'] == 'tensorflow':
        logging.info('Examples with visualizations are only implemented '
                     'with numpy backend.\n'
                     'To change backend, write: '
                     'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
