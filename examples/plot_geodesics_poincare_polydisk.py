"""Plot a geodesic on the Poincare polydisk.

Plot a geodesic on the Poincare polydisk,
with Poincare Disk visualization.
"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.poincare_polydisk import PoincarePolydisk

N_DISKS = 4

POINCARE_POLYDISK = PoincarePolydisk(n_disks=N_DISKS, coords_type='extrinsic')
METRIC = POINCARE_POLYDISK.metric


def plot_geodesic_between_two_points(
        initial_point, end_point, n_steps=11, ax=None):
    """Plot the geodesic between two points."""
    geodesic = METRIC.geodesic(
        initial_point=initial_point, end_point=end_point)
    t = gs.linspace(0., 1., n_steps)
    points = geodesic(t)
    visualization.plot(points, ax=ax, space='poincare_polydisk')


def plot_geodesic_with_initial_tangent_vector(
        initial_point, initial_tangent_vec, n_steps=11, ax=None):
    """Plot the geodesic with initial speed the tangent vector."""
    geodesic = METRIC.geodesic(
        initial_point=initial_point,
        initial_tangent_vec=initial_tangent_vec)
    t = gs.linspace(0., 1., n_steps)
    points = geodesic(t)
    visualization.plot(points, ax=ax, space='poincare_polydisk')


def main():
    """Plot the geodesics."""
    initial_point = gs.array([gs.sqrt(2.), 1., 0.])
    stack_initial_point = gs.stack([initial_point] * N_DISKS, axis=0)
    initial_point = gs.to_ndarray(stack_initial_point, to_ndim=3)

    end_point_intrinsic = gs.array([1.5, 1.5])
    end_point_intrinsic = gs.reshape(end_point_intrinsic, (1, 1, 2))
    end_point = POINCARE_POLYDISK.intrinsic_to_extrinsic_coords(
        end_point_intrinsic)
    end_point = gs.concatenate([end_point] * N_DISKS, axis=1)

    vector = gs.array([3.5, 0.6, 0.8])
    stack_vector = gs.stack([vector] * N_DISKS, axis=0)
    vector = gs.to_ndarray(stack_vector, to_ndim=3)
    initial_tangent_vec = POINCARE_POLYDISK.to_tangent(
        vector=vector, base_point=initial_point)
    fig = plt.figure()
    plot_geodesic_between_two_points(
        initial_point=initial_point, end_point=end_point, ax=fig)
    plot_geodesic_with_initial_tangent_vector(
        initial_point=initial_point,
        initial_tangent_vec=initial_tangent_vec,
        ax=fig)
    plt.show()


if __name__ == '__main__':
    main()
