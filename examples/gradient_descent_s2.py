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

from geomstats.hypersphere import Hypersphere
from geomstats.spd_matrices_space import SPDMatricesSpace
import geomstats.vectorization as vectorization
import geomstats.visualization as visualization


SPHERE2 = Hypersphere(dimension=2)
METRIC = SPHERE2.metric


def gradient_descent(start,
                     loss,
                     grad,
                     manifold,
                     lr=0.5,
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
        x = manifold.metric.exp(base_point=x, tangent_vec=tangent_vec)
        if np.abs(loss(x) - loss(x_prev)) <= precision:
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


def main(video_file='out.mp4', max_iter=128):
    np.random.seed(1983)
    A = generate_well_behaved_matrix()
    loss = lambda x: np.matmul(x.T, np.matmul(A, x))  # NOQA
    grad = lambda x: 2 * np.matmul(A, x)  # NOQA
    initial_point = np.array([0., 1., 0.])
    previous_x = initial_point
    geodesics = []
    n_steps = 20
    # TODO(johmathe): auto differentiation
    # TODO(johmathe): gpu implementation
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
    plot_and_save_video(geodesics, loss, out=video_file, )
    eig, _ = np.linalg.eig(A)
    np.testing.assert_almost_equal(loss(x), np.min(eig), decimal=2)


if __name__ == "__main__":
    main()
