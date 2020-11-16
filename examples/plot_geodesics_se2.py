"""Plot a geodesic of SE(2).

SE2 is equipped with both the canonical Cartan-Shouten connection,
whose geodesics are the group geodesics, i.e. one parameter subgroups and the
left invariant canonical metric.
"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean

SE2_GROUP = SpecialEuclidean(n=2, point_type='matrix')
N_STEPS = 40
METRIC = SE2_GROUP.left_canonical_metric


def main():
    """Plot a group geodesic on SE2."""
    theta = gs.pi / 3
    initial_tangent_vec = gs.array([
        [0., - theta, 2.],
        [theta, 0., 2.],
        [0., 0., 0.]])
    t = gs.linspace(-3., 3., N_STEPS + 1)
    tangent_vec = gs.einsum('t,ij->tij', t, initial_tangent_vec)
    group_geo_points = SE2_GROUP.exp(tangent_vec)
    left_geo_points = METRIC.exp(tangent_vec)

    ax = visualization.plot(
        group_geo_points, space='SE2_GROUP', color='black',
        label='Group Geodesics')
    ax = visualization.plot(
        left_geo_points, ax=ax, space='SE2_GROUP', color='green',
        label='Left Geodesics')
    ax.set_aspect('equal')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()

