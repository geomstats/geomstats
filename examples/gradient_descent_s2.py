"""
Gradient descent on a sphere.

We solve the following optimization problem:

    minimize: x^{T}Ax
    such than: x^{T}x = 1

Using by operating a gradient descent of the quadratic form
on the sphere. We solve this in dimension 3 on the 2-sphere
manifold so that we can visualize and render the path as a video.

To run this example, you need to install ffmpeg:
    pip3 install ffmpeg
"""

import logging

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices

matplotlib.use('Agg')  # NOQA
SPHERE2 = Hypersphere(dim=2)
METRIC = SPHERE2.metric


def gradient_descent(start,
                     loss,
                     grad,
                     manifold,
                     lr=0.01,
                     max_iter=256,
                     precision=1e-5):
    """Operate a gradient descent on a given manifold.

    Until either max_iter or a given precision is reached.
    """
    x = start
    for i in range(max_iter):
        x_prev = x
        euclidean_grad = - lr * grad(x)
        tangent_vec = manifold.to_tangent(
            vector=euclidean_grad, base_point=x)
        x = manifold.metric.exp(base_point=x, tangent_vec=tangent_vec)
        if (gs.abs(loss(x, use_gs=True) - loss(x_prev, use_gs=True))
                <= precision):
            logging.info('x: %s', x)
            logging.info('reached precision %s', precision)
            logging.info('iterations: %d', i)
            break
        yield x, loss(x)


def plot_and_save_video(geodesics,
                        loss,
                        size=20,
                        fps=10,
                        dpi=100,
                        out='out.mp4',
                        color='red'):
    """Render a set of geodesics and save it to an mpeg 4 file."""
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=fps)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection='3d')
    sphere = visualization.Sphere()
    sphere.plot_heatmap(ax, loss)
    points = gs.to_ndarray(geodesics[0], to_ndim=2)
    sphere.add_points(points)
    sphere.draw(ax, color=color, marker='.')
    with writer.saving(fig, out, dpi=dpi):
        for points in geodesics[1:]:
            points = gs.to_ndarray(points, to_ndim=2)
            sphere.draw_points(ax, points=points, color=color, marker='.')
            writer.grab_frame()


def generate_well_behaved_matrix():
    """Generate a matrix with real eigenvalues."""
    matrix = 2 * SPDMatrices(n=3).random_point()
    return matrix


def main(output_file='out.mp4', max_iter=128):
    """Run gradient descent on a sphere."""
    gs.random.seed(1985)
    A = generate_well_behaved_matrix()

    def grad(x):
        return 2 * gs.matmul(A, x)

    def loss(x, use_gs=False):
        if use_gs:
            return gs.matmul(x, gs.matmul(A, x))
        return np.matmul(x, np.matmul(A, x))

    initial_point = gs.array([0., 1., 0.])
    previous_x = initial_point
    geodesics = []
    n_steps = 20
    for x, _ in gradient_descent(initial_point,
                                 loss,
                                 grad,
                                 max_iter=max_iter,
                                 manifold=SPHERE2):
        initial_tangent_vec = METRIC.log(point=x, base_point=previous_x)
        geodesic = METRIC.geodesic(initial_point=previous_x,
                                   initial_tangent_vec=initial_tangent_vec)

        t = np.linspace(0, 1, n_steps)
        geodesics.append(geodesic(t))
        previous_x = x
    if output_file:
        plot_and_save_video(geodesics, loss, out=output_file)
    eig, _ = np.linalg.eig(A)
    np.testing.assert_almost_equal(loss(x), np.min(eig), decimal=2)


if __name__ == '__main__':
    main()
