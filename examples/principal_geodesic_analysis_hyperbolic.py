import matplotlib.pyplot as plt
import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.learning.pca import ExactPGA


def plot_principal_geodesics(points, coords_type):
    space = Hyperbolic(2, coords_type)
    pca = ExactPGA(space)
    pca.fit(points)
    axis1, axis2 = pca.components_
    mean = pca.mean_
    points_proj = pca.fit_transform(points)

    plot_axis1 = gs.stack((mean, mean + axis1))
    plot_axis2 = gs.stack((mean, mean + axis2))

    if coords_type in ["half-space", "ball"]:
        xlim = [-2., 2.] if coords_type == "half-space" else [-1., 1.]
        ylim = [0., 3.] if coords_type == "half-space" else [-1., 1.]

        plt.plot(points[:, 0], points[:, 1], 'o')
        plt.plot(mean[0], mean[1], 'or')
        plt.plot(points_proj[0, :, 0], points_proj[0, :, 1], 'og')
        plt.plot(points_proj[1, :, 0], points_proj[1, :, 1], 'ob')
        plt.plot(plot_axis1[:, 0], plot_axis1[:, 1], 'g')
        plt.plot(plot_axis2[:, 0], plot_axis2[:, 1], 'b')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
    else:
        ax = plt.axes(projection='3d')
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o')
        ax.plot(mean[0], mean[1], mean[2], 'or')
        ax.plot(points_proj[0, :, 0], points_proj[0, :, 1], points_proj[0, :, 2], 'og')
        ax.plot(points_proj[1, :, 0], points_proj[1, :, 1], points_proj[1, :, 2], 'ob')
        ax.plot(plot_axis1[:, 0], plot_axis1[:, 1], plot_axis1[:, 2], 'g')
        ax.plot(plot_axis2[:, 0], plot_axis2[:, 1], plot_axis2[:, 2], 'b')
        plt.show()


space = Hyperbolic(2, coords_type="half-space")
n_points = 100
points = space.random_point(n_points)
points_ext = space.to_coordinates(points, "extrinsic")
points_ball = space.to_coordinates(points, "ball")

plot_principal_geodesics(points, "half-space")
plot_principal_geodesics(points_ball, "ball")
plot_principal_geodesics(points_ext, "extrinsic")
