"""Plot a grid on H2 with Poincare Disk visualization."""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hyperboloid import Hyperboloid

H2 = Hyperboloid(dim=2)
METRIC = H2.metric


def main(left=-4.0, right=4.0, bottom=-4.0, top=4.0, grid_size=32, n_steps=512):
    """Plot a grid on H2 with Poincare Disk visualization.

    Parameters
    ----------
    left, right, bottom, top : ints
        Grid's coordinates
    grid_size : int
        Grid's size.
    n_steps : int
        Number of steps along the geodesics defining the grid.
    """
    starts = []
    ends = []
    for p in gs.linspace(left, right, grid_size):
        starts.append(gs.array([top, p]))
        ends.append(gs.array([bottom, p]))
    for p in gs.linspace(top, bottom, grid_size):
        starts.append(gs.array([p, left]))
        ends.append(gs.array([p, right]))
    starts = [H2.from_coordinates(s, "intrinsic") for s in starts]
    ends = [H2.from_coordinates(e, "intrinsic") for e in ends]
    ax = plt.gca()
    for start, end in zip(starts, ends):
        geodesic = METRIC.geodesic(initial_point=start, end_point=end)

        t = gs.linspace(0.0, 1.0, n_steps)
        points_to_plot = geodesic(t)
        visualization.plot(
            points_to_plot, ax=ax, space="H2_poincare_disk", marker=".", s=1
        )
    plt.show()


if __name__ == "__main__":
    main()
