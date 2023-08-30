"""Plot a geodesic on the Poincare polydisk.

Plot a geodesic on the Poincare polydisk,
with Poincare Disk visualization.
"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.poincare_polydisk import PoincarePolydisk


def plot_geodesic_between_two_points(
    space, initial_point, end_point, n_steps=11, ax=None
):
    """Plot the geodesic between two points."""
    geodesic = space.metric.geodesic(initial_point=initial_point, end_point=end_point)
    t = gs.linspace(0.0, 1.0, n_steps)
    points = geodesic(t)
    visualization.plot(points, ax=ax, space="poincare_polydisk")


def plot_geodesic_with_initial_tangent_vector(
    space, initial_point, initial_tangent_vec, n_steps=11, ax=None
):
    """Plot the geodesic with initial speed the tangent vector."""
    geodesic = space.metric.geodesic(
        initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
    )
    t = gs.linspace(0.0, 1.0, n_steps)
    points = geodesic(t)
    visualization.plot(points, ax=ax, space="poincare_polydisk")


def main():
    """Plot the geodesics."""
    space = PoincarePolydisk(n_disks=4)

    initial_point = gs.array([gs.sqrt(2.0), 1.0, 0.0])
    stack_initial_point = gs.stack([initial_point] * space.n_disks, axis=0)
    initial_point = gs.to_ndarray(stack_initial_point, to_ndim=3)

    end_point_intrinsic = gs.array([1.5, 1.5])
    end_point_intrinsic = gs.reshape(end_point_intrinsic, (1, 1, 2))
    end_point = space.intrinsic_to_extrinsic_coords(end_point_intrinsic)
    end_point = gs.concatenate([end_point] * space.n_disks, axis=1)

    vector = gs.array([3.5, 0.6, 0.8])
    stack_vector = gs.stack([vector] * space.n_disks, axis=0)
    vector = gs.to_ndarray(stack_vector, to_ndim=3)
    initial_tangent_vec = space.to_tangent(vector=vector, base_point=initial_point)
    fig = plt.figure()
    plot_geodesic_between_two_points(
        space, initial_point=initial_point, end_point=end_point, ax=fig
    )
    plot_geodesic_with_initial_tangent_vector(
        space,
        initial_point=initial_point,
        initial_tangent_vec=initial_tangent_vec,
        ax=fig,
    )
    plt.show()


if __name__ == "__main__":
    main()
