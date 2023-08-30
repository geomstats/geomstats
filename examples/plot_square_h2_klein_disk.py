"""Plot a square on H2 with Klein Disk visualization."""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hyperboloid import Hyperboloid


def main():
    """Plot a square on H2 with Klein Disk visualization."""
    square_size = 10

    h2 = Hyperboloid(dim=2)

    top = square_size / 2.0
    bot = -square_size / 2.0
    left = -square_size / 2.0
    right = square_size / 2.0
    corners_int = gs.array([[bot, left], [bot, right], [top, right], [top, left]])
    corners_ext = h2.from_coordinates(corners_int, "intrinsic")
    n_steps = 20
    ax = plt.gca()
    for i, src in enumerate(corners_ext):
        dst_id = (i + 1) % len(corners_ext)
        dst = corners_ext[dst_id]
        tangent_vec = h2.metric.log(point=dst, base_point=src)
        geodesic = h2.metric.geodesic(
            initial_point=src, initial_tangent_vec=tangent_vec
        )
        t = gs.linspace(0.0, 1.0, n_steps)
        edge_points = geodesic(t)

        visualization.plot(
            edge_points, ax=ax, space="H2_klein_disk", marker=".", color="black"
        )
    plt.show()


if __name__ == "__main__":
    main()
