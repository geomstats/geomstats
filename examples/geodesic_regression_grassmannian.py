r"""Compute a geodesic regression on Grassmann manifold (2, 3).

The generative model of the data is:
:math:`Z = Exp_{\beta_0}(\beta_1.X)` and :math:`Y = Exp_Z(\epsilon)`
where:
- :math:`Exp` denotes the Riemannian exponential,
- :math:`\beta_0` is called the intercept,
- :math:`\beta_1` is called the coefficient,
- :math:`\epsilon \sim N(0, 1)` is a standard Gaussian noise,
- :math:`X` is called the input, :math:`Y` is called the y.
"""

import geomstats.backend as gs
from geomstats.geometry.grassmannian import GeneralLinear, Grassmannian
from geomstats.learning.geodesic_regression import GeodesicRegression

SPACE = Grassmannian(3, 2)
METRIC = SPACE.metric
gs.random.seed(0)


def main():
    r"""Compute a geodesic regression on Grassmann manifold (2, 3).

    The generative model of the data is:
    :math:`Z = Exp_{\beta_0}(\beta_1.X)` and :math:`Y = Exp_Z(\epsilon)`
    where:
    - :math:`Exp` denotes the Riemannian exponential,
    - :math:`\beta_0` is called the intercept,
    - :math:`\beta_1` is called the coefficient,
    - :math:`\epsilon \sim N(0, 1)` is a standard Gaussian noise,
    - :math:`X` is called the input, :math:`Y` is called the y.
    """
    # Generate data
    n_samples = 10
    data = gs.random.rand(n_samples)
    data -= gs.mean(data)

    intercept = SPACE.random_uniform()
    beta = SPACE.to_tangent(GeneralLinear(3).random_point(), intercept)
    target = METRIC.exp(
        tangent_vec=gs.einsum("...,jk->...jk", data, beta), base_point=intercept
    )

    # Fit geodesic regression
    gr = GeodesicRegression(
        SPACE,
        metric=METRIC,
        center_X=False,
        method="riemannian",
        max_iter=50,
        init_step_size=0.1,
        verbose=True,
    )

    gr.fit(data, target, compute_training_score=True)
    intercept_hat, beta_hat = gr.intercept_, gr.coef_

    # Measure Mean Squared Error
    mse_intercept = METRIC.squared_dist(intercept_hat, intercept)
    mse_beta = METRIC.squared_norm(
        METRIC.parallel_transport(
            beta_hat, METRIC.log(intercept_hat, intercept), intercept_hat
        )
        - beta,
        intercept,
    )

    # Measure goodness of fit
    r2_hat = gr.training_score_

    print(f"MSE on the intercept: {mse_intercept:.2e}")
    print(f"MSE on the initial velocity beta: {mse_beta:.2e}")
    print(f"Determination coefficient: R^2={r2_hat:.2f}")


if __name__ == "__main__":
    main()
