import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean, variance
from geomstats.learning.geodesic_regression import GeodesicRegression

space = Hypersphere(2)
metric = space.metric
gs.random.seed(0)

# Generate data
n_samples = 50
data = gs.random.rand(n_samples)
data -= gs.mean(data)

intercept = space.random_uniform()
beta = space.to_tangent(5. * gs.random.rand(3), intercept)
target = metric.exp(data[:, None] * beta, intercept)

normal_noise = gs.random.normal(size=(n_samples, 3))
noise = space.to_tangent(normal_noise, target) / 10

rss = gs.sum(metric.squared_norm(noise, target)) / n_samples

# Add some noise
target = metric.exp(noise, target)

# True noise level and R2
estimator = FrechetMean(metric)
estimator.fit(target)
variance_ = variance(target, estimator.estimate_, metric=metric)
r2 = 1 - rss / variance_


# def model(x, base_point, tangent_vec):
#     return metric.exp(x[:, None] * tangent_vec[None], base_point)
#
#
# def loss(x, y, parameter):
#     p, v = gs.split(parameter, 2)
#     base_point = space.projection(p)
#     tangent_vec = space.to_tangent(v, base_point)
#     return 1. / 2. * gs.sum(
#         metric.squared_dist(model(x, base_point, tangent_vec), y))
#
#
# initial_guess = gs.random.rand(6)
#
# # Minimize the loss function given the data
# objective_with_grad = gs.autograd.value_and_grad(
#     lambda param: loss(data, target, param))
# res = minimize(
#     objective_with_grad, initial_guess, method='CG', jac=True,
#     options={'disp': False, 'maxiter': 100})
#
# # Clean result
# intercept_hat, beta_hat = gs.split(gs.array(res.x), 2)
# intercept_hat = space.projection(intercept_hat)
# beta_hat = space.to_tangent(beta_hat, intercept_hat)

gr = GeodesicRegression(space)
intercept_hat, beta_hat, variance_hat = gr.fit(data, target)

# Measure Mean Squared Error
mse_intercept = metric.squared_dist(intercept_hat, intercept)
mse_beta = metric.squared_norm(
    metric.parallel_transport(beta_hat, metric.log(intercept_hat, intercept),
                              intercept_hat) - beta, intercept)

# Measure goodness of fit
r2_hat = 1 - variance_hat / variance_
gr.score(data, target)

print(f'MSE on the intercept: {mse_intercept:.2e}')
print(f'MSE on the initial velocity beta: {mse_beta:.2e}')
print(f'Determination coefficient: R^2={r2_hat:.2f}')

# Plot
fitted_data = gr.predict(data)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
sphere_visu = visualization.Sphere(n_meridians=30)
ax = sphere_visu.set_ax(ax=ax)

path = metric.geodesic(
    initial_point=intercept_hat, initial_tangent_vec=beta_hat)
regressed_geodesic = path(
    gs.linspace(0., 1., 100) * gs.pi * 2 / metric.norm(beta))

i = 10
sphere_visu.draw_points(ax, gs.array([intercept_hat]), marker='o', c='r', s=i)
sphere_visu.draw_points(ax, target, marker='o', c='b', s=i)
sphere_visu.draw_points(ax, fitted_data, marker='o', c='g', s=i)
sphere_visu.draw_points(
    ax, gs.array([intercept]), marker='x', c='r', s=i)

ax.plot(
    regressed_geodesic[:, 0],
    regressed_geodesic[:, 1],
    regressed_geodesic[:, 2], c='gray')
sphere_visu.draw(ax, linewidth=1)
plt.show()
