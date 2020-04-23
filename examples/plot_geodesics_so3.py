"""Plot a geodesic of SO(3).

SO(3) is equipped with its left-invariant canonical METRIC.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import geomstats.visualization as visualization
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

SO3_GROUP = SpecialOrthogonal(n=3, point_type='vector')
METRIC = SO3_GROUP.bi_invariant_metric


def main():
    """Plot a geodesic on SO(3)."""
    initial_point = SO3_GROUP.identity
    initial_tangent_vec = [0.5, 0.5, 0.8]
    geodesic = METRIC.geodesic(initial_point=initial_point,
                               initial_tangent_vec=initial_tangent_vec)

    n_steps = 10
    t = np.linspace(0, 1, n_steps)

    points = geodesic(t)
    visualization.plot(points, space='SO3_GROUP')
    plt.show()


if __name__ == '__main__':
    if os.environ['GEOMSTATS_BACKEND'] == 'tensorflow':
        logging.info('Examples with visualizations are only implemented '
                     'with numpy backend.\n'
                     'To change backend, write: '
                     'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
