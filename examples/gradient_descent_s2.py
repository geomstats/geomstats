"""
Gradient descent on a sphere.

We solve the following optimization problem:

    minimize: x^{T}Ax
    such than: x^{T}x = 1

Using by operating a gradient descent of the quadratic form
on the sphere. We solve this in 3 dimension on the 2-sphere
manifold so that we can visualize and render the path as a video.
"""

import matplotlib
matplotlib.use("Agg")  # NOQA
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['GEOMSTATS_BACKEND'] = 'pytorch'  # NOQA
import geomstats.backend as gs
import geomstats.vectorization as vectorization
import geomstats.visualization as visualization

from geomstats.hypersphere import Hypersphere
from geomstats.spd_matrices_space import SPDMatricesSpace


SPHERE2 = Hypersphere(dimension=2)
METRIC = SPHERE2.metric


def gradient_descent(start,
                     loss,
                     grad,
                     manifold,
                     lr=0.1,
                     max_iter=128,
                     precision=1e-5):
    """Operate a gradient descent on a given manifold until either max_iter or
    a given precision is reached."""
    x = start
    for i in range(max_iter):
        x_prev = x
        euclidean_grad = - lr * grad(x)
        tangent_vec = manifold.projection_to_tangent_space(
                vector=euclidean_grad, base_point=x)
        x = manifold.metric.exp(base_point=x, tangent_vec=tangent_vec)[0]
        if (gs.abs(loss(x, use_gs=True) - loss(x_prev, use_gs=True))
                <= precision):
            print('x: %s' % x)
            print('reached precision %s' % precision)
            print('iterations: %d' % i)
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
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    sphere = visualization.Sphere()
    sphere.plot_heatmap(ax, loss)
    points = vectorization.to_ndarray(geodesics[0], to_ndim=2)
    sphere.add_points(points)
    sphere.draw(ax, color=color, marker='.')
    with writer.saving(fig, out, dpi=dpi):
        for points in geodesics[1:]:
            points = vectorization.to_ndarray(points, to_ndim=2)
            sphere.add_points(points)
            sphere.draw_points(ax, color=color, marker='.')
            writer.grab_frame()


def generate_well_behaved_matrix():
    """Generate a matrix with real eigenvalues."""
    matrix = 2 * SPDMatricesSpace(n=3).random_uniform()[0]
    assert np.linalg.det(matrix) > 0
    return matrix


def main(output_file='out.mp4', max_iter=128):
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
    for x, fx in gradient_descent(initial_point,
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


if __name__ == "__main__":
    main()
