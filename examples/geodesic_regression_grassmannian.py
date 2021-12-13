import os

os.environ["GEOMSTATS_BACKEND"] = "tensorflow"

import geomstats.backend as gs
from geomstats.geometry.grassmannian import GeneralLinear, Grassmannian
from geomstats.learning.frechet_mean import FrechetMean, variance
from geomstats.learning.geodesic_regression import GeodesicRegression

space = Grassmannian(3, 2)
metric = space.metric
gs.random.seed(0)
p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

# Generate data
n_samples = 10
data = gs.random.rand(n_samples)
data -= gs.mean(data)

intercept = space.random_uniform()
beta = space.to_tangent(GeneralLinear(3).random_point(), intercept)
target = metric.exp(
    tangent_vec=gs.einsum("...,jk->...jk", data, beta), base_point=intercept
)

# True variance
estimator = FrechetMean(metric)
estimator.fit(target)
variance_ = variance(target, estimator.estimate_, metric=metric)

gr = GeodesicRegression(
    space,
    metric=metric,
    center_X=False,
    method="riemannian",
    verbose=True,
    max_iter=50,
    learning_rate=0.1,
)
gr.fit(data, target, compute_training_score=True)
intercept_hat, beta_hat = gr.intercept_, gr.coef_

# Measure Mean Squared Error
mse_intercept = metric.squared_dist(intercept_hat, intercept)
mse_beta = metric.squared_norm(
    metric.parallel_transport(
        beta_hat, metric.log(intercept_hat, intercept), intercept_hat
    )
    - beta,
    intercept,
)

# Measure goodness of fit
r2_hat = gr.training_score_

print(f"MSE on the intercept: {mse_intercept:.2e}")
print(f"MSE on the initial velocity beta: {mse_beta:.2e}")
print(f"Determination coefficient: R^2={r2_hat:.2f}")
