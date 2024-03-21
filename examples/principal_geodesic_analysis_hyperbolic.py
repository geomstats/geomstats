import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.learning.pca import ExactPGA


def plot_principal_geodesics(points, coords_type):
    """Compute and plot principal geodesics."""
    space = Hyperbolic(2, coords_type)
    pca = ExactPGA(space)
    pca.fit(points)
    vec_1, vec_2 = pca.components_
    mean = pca.mean_
    points_proj = pca.fit_transform(points)

    axis_1 = gs.stack((mean, mean + vec_1))
    axis_2 = gs.stack((mean, mean + vec_2))
    angles = gs.linspace(0.0, 2 * gs.pi, 100)
    circle = gs.stack((gs.cos(angles), gs.sin(angles)))

    if coords_type in ["half-space", "ball"]:
        xlim = [-2.0, 2.0] if coords_type == "half-space" else [-1.0, 1.0]
        ylim = [0.0, 3.0] if coords_type == "half-space" else [-1.0, 1.0]

        plt.plot(points[:, 0], points[:, 1], "o")
        plt.plot(mean[0], mean[1], "or")
        plt.plot(points_proj[0, :, 0], points_proj[0, :, 1], "og")
        plt.plot(points_proj[1, :, 0], points_proj[1, :, 1], "ob")
        plt.plot(axis_1[:, 0], axis_1[:, 1], "g")
        plt.plot(axis_2[:, 0], axis_2[:, 1], "b")
        if coords_type == "ball":
            plt.plot(circle[0], circle[1])
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
    else:
        ax = plt.axes(projection="3d")
        ax.plot(points[:, 0], points[:, 1], points[:, 2], "o")
        ax.plot(mean[0], mean[1], mean[2], "or")
        ax.plot(points_proj[0, :, 0], points_proj[0, :, 1], points_proj[0, :, 2], "og")
        ax.plot(points_proj[1, :, 0], points_proj[1, :, 1], points_proj[1, :, 2], "ob")
        ax.plot(axis_1[:, 0], axis_1[:, 1], axis_1[:, 2], "g")
        ax.plot(axis_2[:, 0], axis_2[:, 1], axis_2[:, 2], "b")
        plt.show()


def main():
    """Exact Principal Geodesic Analysis in hyperbolic plane.

    Perform exact Principal Geodesic Analysis on random data points
    on three different models of two-dimensional hyperbolic geometry:
    - the Poincare half-space
    - the Poincare ball
    - the hyperboloid
    and visualize the principal axes and geodesics.
    """
    space = Hyperbolic(2, coords_type="half-space")
    n_points = 100
    points = space.random_point(n_points)
    points_ext = space.to_coordinates(points, "extrinsic")
    points_ball = space.to_coordinates(points, "ball")

    plot_principal_geodesics(points, "half-space")
    plot_principal_geodesics(points_ball, "ball")
    plot_principal_geodesics(points_ext, "extrinsic")


if __name__ == "__main__":
    main()
