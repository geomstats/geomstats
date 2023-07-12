"""Plot a geodesic of SE(3).

SE3 is equipped with its left-invariant canonical metric.
"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean


def main():
    """Plot a geodesic on SE3."""
    n_steps = 40

    se3_group = SpecialEuclidean(n=3, point_type="vector")

    initial_point = se3_group.identity
    initial_tangent_vec = gs.array([1.8, 0.2, 0.3, 3.0, 3.0, 1.0])
    geodesic = se3_group.metric.geodesic(
        initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
    )

    t = gs.linspace(-3.0, 3.0, n_steps)

    points = geodesic(t)

    visualization.plot(points, space="SE3_GROUP")
    plt.show()


if __name__ == "__main__":
    main()
