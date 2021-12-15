import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.model_selection import train_test_split
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean, variance
from geomstats.learning.geodesic_regression import GeodesicRegression
from geomstats.learning.wrapped_gaussian_process import WrappedGaussianProcess

# Define the space and metric
DIM = 2
SPACE = Hypersphere(dim=DIM)
EMBEDDING_DIM = SPACE.embedding_space.dim
METRIC = SPACE.metric
gs.random.seed(0)

# Generate sinusoidal data on the sphere
# First generate the geodesic
n_samples = 100
X = gs.linspace(0, 1.5 * np.pi, n_samples)
Xs = gs.linspace(0, 1.5 * np.pi, n_samples * 10).reshape(-1, 1)
intercept = np.array([0, -1, 0])
coef = np.array([1, 0, 0.5])
y = METRIC.exp(X[:, None] * coef, base_point=intercept)
# Then add orthogonal sinusoidal oscillations
o = 1 / 20 * np.array([-0.5, 0, 1])
o = SPACE.to_tangent(o, base_point=y)
s = X[:, None] * np.sin(5 * np.pi * X[:, None])
y = METRIC.exp(s * o, base_point=y)

# Add noise
# normal_noise = gs.random.normal(size=(n_samples, EMBEDDING_DIM))
# noise = SPACE.to_tangent(normal_noise, base_point=y) / gs.pi / 2
# y3 = METRIC.exp(noise, y2)

fig = plt.figure(figsize=(8, 8))
ax = visualization.plot(y, space='S2', color='black', alpha=0.7, label='Data points')
ax.set_box_aspect([1, 1, 1])
ax.legend()
plt.show()

# Wrapped Gaussian Process Regression
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.6, random_state=42)
prior = lambda X: METRIC.exp(X * coef, base_point=intercept)
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

fig = plt.figure(figsize=(8, 8))
ax = visualization.plot(y_train, space='S2', color='black', alpha=0.7, label='Train data points')
ax.scatter(prior(X_train)[:, 0], prior(X_train)[:, 1], prior(X_train)[:, 2], color='red', alpha=.7, label='prior')
ax.set_box_aspect([1, 1, 1])
ax.legend()
plt.show()

wgp = WrappedGaussianProcess(space=SPACE, metric=METRIC, prior=prior, kernel=kernel)
wgp.fit(X_train, y_train)
f_pred = wgp.predict(Xs)

fig = plt.figure(figsize=(8, 8))
ax = visualization.plot(y_train, space='S2', color='black', alpha=0.7, label='Train data points')
ax.plot(f_pred[:, 0], f_pred[:, 1], f_pred[:, 2], color='blue', alpha=.7, label='f prediction')
ax.set_box_aspect([1, 1, 1])
ax.legend()
plt.show()

# y1d = s + noise[:, 1].reshape(-1, 1)
# plt.scatter(X, y1d, color='red')
# plt.scatter(X, s.reshape(-1, 1), color='blue')
# plt.show()
#
# X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.5, random_state=42)
#
# wgp = WrappedGaussianProcess(space=SPACE, metric=METRIC, prior=METRIC.exp(X*coef, base_point=intercept))
# wgp.fit(X_train, y_train)
