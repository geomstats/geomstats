"""
Gradient descent on a sphere.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # NOQA
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from geomstats.hypersphere import Hypersphere
import geomstats.visualization as visualization
import geomstats.vectorization as vectorization

SPHERE2 = Hypersphere(dimension=2)
METRIC = SPHERE2.metric


def gradient_descent(start,
                     loss,
                     grad,
                     manifold,
                     lr=0.05,
                     max_iter=128,
                     precision=1e-5):
    x = start
    for i in range(max_iter):
        x_prev = x
        euclidian = - lr * grad(x)
        tangent_vec = manifold.projection_to_tangent_space(vector=euclidian,
                                                           base_point=x)
        x = manifold.metric.exp(base_point=x, tangent_vec=tangent_vec)[0]
        if np.abs(loss(x) - loss(x_prev)) <= precision:
            print('x: %s' % x)
            print('reached precision %s' % precision)
            print('iterations: %d' % i)
            break
        yield x, loss(x)


def animate(geodesics, size=20, fps=0, dpi=103, out='out.mp4', color='red'):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    sphere = visualization.Sphere()
    points = vectorization.to_ndarray(geodesics[0], to_ndim=2)
    sphere.add_points(points)
    sphere.draw(ax, color=color, marker='.')
    with writer.saving(fig, 'foo.mp4', dpi=dpi):
        for points in geodesics[1:]:
            points = vectorization.to_ndarray(points, to_ndim=2)
            sphere.add_points(points)
            sphere.draw_points(ax, color=color, marker='.')
            writer.grab_frame()


def main():
    np.random.seed(1980)
    A = np.random.random(size=(3, 3))
    loss = lambda x: x.T @ A @ x  # NOQA
    grad = lambda x: 2 * A @ x  # NOQA
    initial_point = np.array([0., 1., 0.])
    previous_x = initial_point
    geodesics = []
    n_steps = 20
    # TODO(johmathe): auto differentiation
    # TODO(johmathe): gpu implementation
    for x, fx in gradient_descent(initial_point, loss, grad, manifold=SPHERE2):
        initial_tangent_vec = METRIC.log(point=x, base_point=previous_x)
        geodesic = METRIC.geodesic(initial_point=previous_x,
                                   initial_tangent_vec=initial_tangent_vec)

        t = np.linspace(0, 1, n_steps)
        geodesics.append(geodesic(t))
        previous_x = x
    animate(geodesics)
    eig, _ = np.linalg.eig(A)
    np.testing.assert_almost_equal(loss(x), np.min(eig), decimal=2)


if __name__ == "__main__":
    main()
