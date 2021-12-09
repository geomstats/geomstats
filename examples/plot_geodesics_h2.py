"""Plot a geodesic in H2.

Plot a geodesic on the hyperbolic space H2.
With Poincare Disk visualization.
"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hyperboloid import Hyperboloid

H2 = Hyperboloid(dim=2, coords_type="extrinsic")
METRIC = H2.metric


def plot_geodesic_between_two_points(initial_point, end_point, n_steps=10, ax=None):
    """Plot the geodesic between two points."""
    if not H2.belongs(initial_point):
        raise ValueError("The initial point of the geodesic is not in H2.")
    if not H2.belongs(end_point):
        raise ValueError("The end point of the geodesic is not in H2.")

    geodesic = METRIC.geodesic(initial_point=initial_point, end_point=end_point)

    t = gs.linspace(0.0, 1.0, n_steps)
    points = geodesic(t)
    visualization.plot(points, ax=ax, space="H2_poincare_disk")


def plot_geodesic_with_initial_tangent_vector(
    initial_point, initial_tangent_vec, n_steps=10, ax=None
):
    """Plot the geodesic with initial speed the tangent vector."""
    if not H2.belongs(initial_point):
        raise ValueError("The initial point of the geodesic is not in H2.")
    geodesic = METRIC.geodesic(
        initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
    )

    t = gs.linspace(0.0, 1.0, n_steps)
    points = geodesic(t)
    visualization.plot(points, ax=ax, space="H2_poincare_disk")


def main():
    """Plot geodesics on H2."""
    initial_point = gs.array([gs.sqrt(2.0), 1.0, 0.0])
    end_point = gs.array([1.5, 1.5])
    end_point = H2.from_coordinates(end_point, "intrinsic")
    initial_tangent_vec = H2.to_tangent(
        vector=gs.array([3.5, 0.6, 0.8]), base_point=initial_point
    )

    ax = plt.gca()
    plot_geodesic_between_two_points(initial_point, end_point, ax=ax)
    plot_geodesic_with_initial_tangent_vector(initial_point, initial_tangent_vec, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
