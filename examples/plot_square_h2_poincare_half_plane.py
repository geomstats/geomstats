"""Plot a square on H2 with Poincare half-plane visualization."""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hyperboloid import Hyperboloid

H2 = Hyperboloid(dim=2)
METRIC = H2.metric

SQUARE_SIZE = 10


def main():
    """Plot a square on H2 with Poincare half-plane visualization."""
    top = SQUARE_SIZE / 2.0
    bot = - SQUARE_SIZE / 2.0
    left = - SQUARE_SIZE / 2.0
    right = SQUARE_SIZE / 2.0
    corners_int = gs.array([
        [bot, left], [bot, right], [top, right], [top, left]])
    corners_ext = H2.from_coordinates(corners_int, 'intrinsic')
    n_steps = 20
    ax = plt.gca()
    edge_points = []
    for i, src in enumerate(corners_ext):
        dst_id = (i + 1) % len(corners_ext)
        dst = corners_ext[dst_id]
        geodesic = METRIC.geodesic(initial_point=src, end_point=dst)
        t = gs.linspace(0., 1., n_steps)
        edge_points.append(geodesic(t))

    edge_points = gs.vstack(edge_points)
    visualization.plot(
        edge_points,
        ax=ax,
        space='H2_poincare_half_plane',
        point_type='extrinsic',
        marker='.',
        color='black')

    plt.show()


if __name__ == '__main__':
    main()
