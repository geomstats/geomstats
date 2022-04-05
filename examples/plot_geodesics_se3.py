"""Plot a geodesic of SE(3).

SE3 is equipped with its left-invariant canonical metric.
"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean

SE3_GROUP = SpecialEuclidean(n=3, point_type="vector")
METRIC = SE3_GROUP.left_canonical_metric
N_STEPS = 40


def main():
    """Plot a geodesic on SE3."""
    initial_point = SE3_GROUP.identity
    initial_tangent_vec = gs.array([1.8, 0.2, 0.3, 3.0, 3.0, 1.0])
    geodesic = METRIC.geodesic(
        initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
    )

    t = gs.linspace(-3.0, 3.0, N_STEPS)

    points = geodesic(t)

    visualization.plot(points, space="SE3_GROUP")
    plt.show()


if __name__ == "__main__":
    main()
