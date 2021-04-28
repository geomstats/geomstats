"""Plot a geodesic of SE(2).

SE2 is equipped with the canonical Cartan-Shouten connection,
whose geodesics are the group geodesics, i.e. one parameter subgroups and the
left and right invariant canonical metric. In all cases the rotation part is
the same: it has constant angular velocity. The translation parts differe
however.
"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean

SE2_GROUP = SpecialEuclidean(n=2, point_type='matrix')
N_STEPS = 40


def main():
    """Plot geodesics on SE(2) with different structures."""
    theta = gs.pi / 6
    initial_tangent_vec = gs.array([
        [0., - theta, .5],
        [theta, 0., .5],
        [0., 0., 0.]])
    t = gs.linspace(-2., 2., N_STEPS + 1)
    tangent_vec = gs.einsum('t,ij->tij', t, initial_tangent_vec)
    group_geo_points = SE2_GROUP.exp(tangent_vec)
    geo_points = SE2_GROUP.call_method_on_metrics('exp', tangent_vec)

    ax = visualization.plot(
        group_geo_points, space='SE2_GROUP', color='black',
        label='Group')

    # for (key, value), color in zip(geo_points.items(),
    #                                ['yellow', 'green']):
    #     ax = visualization.plot(
    #         value, ax=ax, space='SE2_GROUP', color=color,
    #         label=key)

    for value, label, color in zip(geo_points.values(),
                                   ['left', 'right'],
                                   ['yellow', 'green']):
        ax = visualization.plot(
            value, ax=ax, space='SE2_GROUP', color=color,
            label=label)

    ax.set_aspect('equal')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
