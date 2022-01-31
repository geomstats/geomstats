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

DIM = 2
SPACE = Hypersphere(dim=DIM)
EMBEDDING_DIM = SPACE.embedding_space.dim
METRIC = SPACE.metric
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
    # Generate noise-free data
    n_samples = 50
    X = gs.random.rand(n_samples)
    X -= gs.mean(X)

    intercept = SPACE.random_uniform()
    coef = SPACE.to_tangent(5.0 * gs.random.rand(EMBEDDING_DIM), base_point=intercept)
    y = METRIC.exp(X[:, None] * coef, base_point=intercept)

    # Generate normal noise
    normal_noise = gs.random.normal(size=(n_samples, EMBEDDING_DIM))
    noise = SPACE.to_tangent(normal_noise, base_point=y) / gs.pi / 2

    rss = gs.sum(METRIC.squared_norm(noise, base_point=y)) / n_samples

    # Add noise
    y = METRIC.exp(noise, y)

    # True noise level and R2
    estimator = FrechetMean(METRIC)
    estimator.fit(y)
    variance_ = variance(y, estimator.estimate_, metric=METRIC)
    r2 = 1 - rss / variance_

    # Fit geodesic regression
    gr = GeodesicRegression(SPACE, center_X=False, method="extrinsic", verbose=True)
    gr.fit(X, y, compute_training_score=True)
    intercept_hat, coef_hat = gr.intercept_, gr.coef_

    # Measure Mean Squared Error
    mse_intercept = METRIC.squared_dist(intercept_hat, intercept)

    tangent_vec_to_transport = coef_hat
    tangent_vec_of_transport = METRIC.log(intercept, base_point=intercept_hat)
    transported_coef_hat = METRIC.parallel_transport(
        tangent_vec=tangent_vec_to_transport,
        base_point=intercept_hat,
        direction=tangent_vec_of_transport,
    )
    mse_coef = METRIC.squared_norm(transported_coef_hat - coef, base_point=intercept)

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

    path = METRIC.geodesic(initial_point=intercept_hat, initial_tangent_vec=coef_hat)
    regressed_geodesic = path(
        gs.linspace(0.0, 1.0, 100) * gs.pi * 2 / METRIC.norm(coef)
    )
    regressed_geodesic = gs.to_numpy(gs.autodiff.detach(regressed_geodesic))

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
