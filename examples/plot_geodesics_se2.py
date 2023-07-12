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
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.special_euclidean import (
    SpecialEuclidean,
    SpecialEuclideanMatrixCanonicalLeftMetric,
)


def main():
    """Plot geodesics on SE(2) with different structures."""
    n_steps = 40

    se2_group_left_metric = SpecialEuclidean(n=2, point_type="matrix", equip=False)
    se2_group_left_metric.equip_with_metric(SpecialEuclideanMatrixCanonicalLeftMetric)

    se2_group_right_metric = SpecialEuclidean(n=2, point_type="matrix", equip=False)
    se2_group_right_metric.equip_with_metric(InvariantMetric, left=False)

    theta = gs.pi / 6
    initial_tangent_vec = gs.array(
        [[0.0, -theta, 0.5], [theta, 0.0, 0.5], [0.0, 0.0, 0.0]]
    )
    t = gs.linspace(-2.0, 2.0, n_steps + 1)
    tangent_vec = gs.einsum("t,ij->tij", t, initial_tangent_vec)
    group_geo_points = se2_group_left_metric.exp(tangent_vec)
    left_geo_points = se2_group_left_metric.metric.exp(tangent_vec)
    right_geo_points = se2_group_right_metric.metric.exp(tangent_vec)

    ax = visualization.plot(
        group_geo_points, space="SE2_GROUP", color="black", label="Group"
    )
    ax = visualization.plot(
        left_geo_points, ax=ax, space="SE2_GROUP", color="yellow", label="Left"
    )
    ax = visualization.plot(
        right_geo_points,
        ax=ax,
        space="SE2_GROUP",
        color="green",
        label="Right by Integration",
    )
    ax.set_aspect("equal")
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
