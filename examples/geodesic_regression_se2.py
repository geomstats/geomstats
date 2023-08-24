r"""Compute and visualize a geodesic regression on SE(2).

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
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.frechet_mean import FrechetMean, variance
from geomstats.learning.geodesic_regression import GeodesicRegression

gs.random.seed(0)


def main():
    r"""Compute and visualize a geodesic regression on the SE(2).

    The generative model of the data is:
    :math:`Z = Exp_{\beta_0}(\beta_1.X)` and :math:`Y = Exp_Z(\epsilon)`
    where:
    - :math:`Exp` denotes the Riemannian exponential,
    - :math:`\beta_0` is called the intercept,
    - :math:`\beta_1` is called the coefficient,
    - :math:`\epsilon \sim N(0, 1)` is a standard Gaussian noise,
    - :math:`X` is the input, :math:`Y` is the target.
    """
    space = SpecialEuclidean(2)

    # Generate noise-free data
    n_samples = 20
    X = gs.random.normal(size=(n_samples,))
    X -= gs.mean(X)

    intercept = space.random_point()
    coef = space.to_tangent(5.0 * gs.random.rand(3, 3), intercept)
    y = space.metric.exp(X[:, None, None] * coef[None], intercept)

    # Generate normal noise in the Lie algebra
    normal_noise = gs.random.normal(size=(n_samples, 3))
    normal_noise = space.lie_algebra.matrix_representation(normal_noise)
    noise = space.tangent_translation_map(y)(normal_noise) / gs.pi

    rss = gs.sum(space.metric.squared_norm(noise, y)) / n_samples

    # Add noise
    y = space.metric.exp(noise, y)

    # True noise level and R2
    estimator = FrechetMean(space)
    estimator.fit(y)
    variance_ = variance(space, y, estimator.estimate_)
    r2 = 1 - rss / variance_

    # Fit geodesic regression
    gr = GeodesicRegression(
        space,
        center_X=False,
        method="riemannian",
        initialization="frechet",
        compute_training_score=True,
    ).set(
        max_iter=100,
        init_step_size=0.1,
        verbose=True,
    )
    gr.fit(X, y)

    intercept_hat, beta_hat = gr.intercept_, gr.coef_

    # Measure Mean Squared Error
    mse_intercept = space.metric.squared_dist(intercept_hat, intercept)
    mse_beta = space.metric.squared_norm(
        space.metric.parallel_transport(
            beta_hat, intercept_hat, space.metric.log(intercept_hat, intercept)
        )
        - coef,
        intercept,
    )

    # Measure goodness of fit
    r2_hat = gr.training_score_

    print(f"MSE on the intercept: {mse_intercept:.2e}")
    print(f"MSE on the initial velocity beta: {mse_beta:.2e}")
    print(f"Determination coefficient: R^2={r2_hat:.2f}")
    print(f"True R^2: {r2:.2f}")

    # Plot
    fitted_data = gr.predict(X)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    sphere_visu = visualization.SpecialEuclidean2()
    ax = sphere_visu.set_ax(ax=ax)

    path = space.metric.geodesic(
        initial_point=intercept_hat, initial_tangent_vec=beta_hat
    )
    regressed_geodesic = path(gs.linspace(min(X), max(X), 100))

    sphere_visu.draw_points(ax, y, marker="o", c="black")
    sphere_visu.draw_points(ax, fitted_data, marker="o", c="gray")
    sphere_visu.draw_points(ax, gs.array([intercept]), marker="x", c="r")
    sphere_visu.draw_points(ax, gs.array([intercept_hat]), marker="o", c="green")

    ax.plot(regressed_geodesic[:, 0, 2], regressed_geodesic[:, 1, 2], c="gray")
    plt.show()


if __name__ == "__main__":
    main()
