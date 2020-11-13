"""Plot a geodesic of SE(2).

SE2 is equipped canonical Cartan-Shouten connection. The geodesics correspond
to one-parameter subgroups.
"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean

SE2_GROUP = SpecialEuclidean(n=2, point_type='matrix')
N_STEPS = 40


def main():
    """Plot a geodesic on SE2."""
    theta = gs.pi / 3
    initial_tangent_vec = gs.array([
        [0., - theta, 2.],
        [theta, 0., 3.],
        [0., 0., 0.]])
    t = gs.linspace(-3., 3., N_STEPS)
    tangent_vec = gs.einsum('t,ij->tij', t, initial_tangent_vec)
    points = SE2_GROUP.exp(tangent_vec)

    ax = visualization.plot(points, space='SE2_GROUP', color='black')
    ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    main()

