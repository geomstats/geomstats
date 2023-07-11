r"""Compute and visualize a geodesic regression on the sphere.

The generative model of the data is:
:math:`Z = Exp_{\beta_0}(\beta_1.X)` and :math:`Y = Exp_Z(\epsilon)`
where:
- :math:`Exp` denotes the Riemannian exponential,
- :math:`\beta_0` is called the intercept,
- :math:`\beta_1` is called the coefficient,
- :math:`\epsilon \sim N(0, 1)` is a standard Gaussian noise,
- :math:`X` is called the input, :math:`Y` is called the y.
"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean, variance
from geomstats.learning.geodesic_regression import GeodesicRegression

gs.random.seed(0)


def main():
    r"""Compute and visualize a geodesic regression on the sphere.

    The generative model of the data is:
    :math:`Z = Exp_{\beta_0}(\beta_1.X)` and :math:`Y = Exp_Z(\epsilon)`
    where:
    - :math:`Exp` denotes the Riemannian exponential,
    - :math:`\beta_0` is called the intercept,
    - :math:`\beta_1` is called the coefficient,
    - :math:`\epsilon \sim N(0, 1)` is a standard Gaussian noise,
    - :math:`X` is the input, :math:`Y` is the target.
    """
    space = Hypersphere(dim=2)

    # Generate noise-free data
    n_samples = 50
    X = gs.random.rand(n_samples)
    X -= gs.mean(X)

    intercept = space.random_uniform()
    coef = space.to_tangent(
        5.0 * gs.random.rand(space.embedding_space.dim), base_point=intercept
    )
    y = space.metric.exp(X[:, None] * coef, base_point=intercept)

    # Generate normal noise
    normal_noise = gs.random.normal(size=(n_samples, space.embedding_space.dim))
    noise = space.to_tangent(normal_noise, base_point=y) / gs.pi / 2

    rss = gs.sum(space.metric.squared_norm(noise, base_point=y)) / n_samples

    # Add noise
    y = space.metric.exp(noise, y)

    # True noise level and R2
    estimator = FrechetMean(space)
    estimator.fit(y)
    variance_ = variance(space, y, estimator.estimate_)
    r2 = 1 - rss / variance_

    # Fit geodesic regression
    gr = GeodesicRegression(
        space, center_X=False, method="extrinsic", compute_training_score=True
    )
    gr.fit(X, y)
    intercept_hat, coef_hat = gr.intercept_, gr.coef_

    # Measure Mean Squared Error
    mse_intercept = space.metric.squared_dist(intercept_hat, intercept)

    tangent_vec_to_transport = coef_hat
    tangent_vec_of_transport = space.metric.log(intercept, base_point=intercept_hat)
    transported_coef_hat = space.metric.parallel_transport(
        tangent_vec=tangent_vec_to_transport,
        base_point=intercept_hat,
        direction=tangent_vec_of_transport,
    )
    mse_coef = space.metric.squared_norm(
        transported_coef_hat - coef, base_point=intercept
    )

    # Measure goodness of fit
    r2_hat = gr.training_score_

    print(f"MSE on the intercept: {mse_intercept:.2e}")
    print(f"MSE on the coef, i.e. initial velocity: {mse_coef:.2e}")
    print(f"Determination coefficient: R^2={r2_hat:.2f}")
    print(f"True R^2: {r2:.2f}")

    # Plot
    fitted_data = gr.predict(X)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    sphere_visu = visualization.Sphere(n_meridians=30)
    ax = sphere_visu.set_ax(ax=ax)

    path = space.metric.geodesic(
        initial_point=intercept_hat, initial_tangent_vec=coef_hat
    )
    regressed_geodesic = path(
        gs.linspace(0.0, 1.0, 100) * gs.pi * 2 / space.metric.norm(coef)
    )
    regressed_geodesic = gs.to_numpy(regressed_geodesic)

    size = 10
    marker = "o"
    sphere_visu.draw_points(ax, gs.array([intercept_hat]), marker=marker, c="r", s=size)
    sphere_visu.draw_points(ax, y, marker=marker, c="b", s=size)
    sphere_visu.draw_points(ax, fitted_data, marker=marker, c="g", s=size)

    ax.plot(
        regressed_geodesic[:, 0],
        regressed_geodesic[:, 1],
        regressed_geodesic[:, 2],
        c="gray",
    )
    sphere_visu.draw(ax, linewidth=1)
    ax.grid(False)
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
