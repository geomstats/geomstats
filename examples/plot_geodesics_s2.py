"""Plot a geodesic on the sphere S2."""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere


def main():
    """Plot a geodesic on the sphere."""
    space = Hypersphere(dim=2)

    initial_point = gs.array([1.0, 0.0, 0.0])
    initial_tangent_vec = space.to_tangent(
        vector=gs.array([1.0, 2.0, 0.8]), base_point=initial_point
    )
    geodesic = space.metric.geodesic(
        initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
    )

    n_steps = 10
    t = gs.linspace(0.0, 1.0, n_steps)

    points = geodesic(t)
    visualization.plot(points, space="S2")
    plt.show()


if __name__ == "__main__":
    main()
