import matplotlib.pyplot as plt
import os
os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.frechet_mean import FrechetMean, variance
from geomstats.learning.geodesic_regression import GeodesicRegression

space = SpecialEuclidean(2)
metric = space.left_canonical_metric
metric.default_point_type = 'matrix'
gs.random.seed(0)

# Generate data
n_samples = 20
data = gs.random.rand(n_samples)
data -= gs.mean(data)

intercept = space.random_uniform()
beta = space.to_tangent(5. * gs.random.rand(3, 3), intercept)
target = metric.exp(data[:, None, None] * beta[None], intercept)

# Generate normal noise
normal_noise = gs.random.normal(size=(n_samples, 3, 3))
noise = space.to_tangent(normal_noise, target) / gs.pi / 2

rss = gs.sum(metric.squared_norm(noise, target)) / n_samples

# Add noise
target = metric.exp(noise, target)

# True noise level and R2
estimator = FrechetMean(metric)
estimator.fit(target)
variance_ = variance(target, estimator.estimate_, metric=metric)
r2 = 1 - rss / variance_

gr = GeodesicRegression(
    space, metric=metric, center_data=False, algorithm='riemannian',
    verbose=True, max_iter=50, learning_rate=0.1)
gr.fit(data, target, compute_training_score=True)
intercept_hat, beta_hat = gr.intercept_, gr.coef_

# Measure Mean Squared Error
mse_intercept = metric.squared_dist(intercept_hat, intercept)
mse_beta = metric.squared_norm(
    metric.parallel_transport(beta_hat, metric.log(intercept_hat, intercept),
                              intercept_hat) - beta, intercept)

# Measure goodness of fit
r2_hat = gr.training_score_

print(f'MSE on the intercept: {mse_intercept:.2e}')
print(f'MSE on the initial velocity beta: {mse_beta:.2e}')
print(f'Determination coefficient: R^2={r2_hat:.2f}')

# Plot
fitted_data = gr.predict(data)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
sphere_visu = visualization.SpecialEuclidean2()
ax = sphere_visu.set_ax(ax=ax)

path = metric.geodesic(
    initial_point=intercept_hat, initial_tangent_vec=beta_hat)
regressed_geodesic = path(
    gs.linspace(0., 1., 100) / metric.norm(beta))

i = 10
sphere_visu.draw_points(ax, gs.array([intercept_hat]), marker='o', c='green')
sphere_visu.draw_points(ax, target, marker='o', c='black')
sphere_visu.draw_points(ax, fitted_data, marker='o', c='gray')
sphere_visu.draw_points(
    ax, gs.array([intercept]), marker='x', c='r')

ax.plot(
    regressed_geodesic[:, 0, 2],
    regressed_geodesic[:, 1, 2], c='gray')
plt.show()
