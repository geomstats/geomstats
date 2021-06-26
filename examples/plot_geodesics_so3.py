"""Plot a geodesic of SO(3).

SO(3) is equipped with its bi-invariant canonical metric.
"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

SO3_GROUP = SpecialOrthogonal(n=3, point_type="vector")
METRIC = SO3_GROUP.bi_invariant_metric
N_STEPS = 10


def main():
    """Plot a geodesic on SO(3)."""
    initial_point = SO3_GROUP.identity
    initial_tangent_vec = gs.array([0.5, 0.5, 0.8])
    geodesic = METRIC.geodesic(
        initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
    )

    t = gs.linspace(0.0, 1.0, N_STEPS)

    points = geodesic(t)
    visualization.plot(points, space="SO3_GROUP")
    plt.show()


if __name__ == "__main__":
    main()
