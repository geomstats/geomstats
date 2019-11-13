"""
Compares the mean of a data set of unit vectors
to the projection of the mean of rotations onto the sphere,
seen as the homogeneous space 3D/2D rotations
"""

import matplotlib.pyplot as plt
import numpy as np

import geomstats.visualization as visualization

from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.geometry.special_orthogonal_group import SpecialOrthogonalGroup

def embed(point):
    """
    embeds 2D rotations in space of 3D rotations
    :param point:
    :return:
    """
    d = np.zeros((point.shape[0], 3, 3)) + np.array([[1, 0, 0], [0,0,0],[0,0,0]])
    for i, p in enumerate(point):
        d[i, 1:, 1:] = point[i, :, :]
    return d

SO3_GROUP = SpecialOrthogonalGroup(n=3)
SO2_GROUP = SpecialOrthogonalGroup(n=2)
S2_SPHERE = Hypersphere(dimension=2)
METRIC = SO3_GROUP.bi_invariant_metric
Sphere_Metric = HypersphereMetric(dimension=2)

N_SAMPLES = 10

data = SO3_GROUP.random_uniform(n_samples=N_SAMPLES)
mean = METRIC.mean(data)
projected_mean = SO3_GROUP.matrix_from_rotation_vector(mean).dot(np.array([1, 0, 0]))
assert S2_SPHERE.belongs(projected_mean)

projected_data = SO3_GROUP.matrix_from_rotation_vector(data).dot(np.array([1, 0, 0]))
assert S2_SPHERE.belongs(projected_data).all()
frechet_mean = Sphere_Metric.mean(projected_data)
print(Sphere_Metric.dist(frechet_mean, projected_mean))

horizontal_lie_algebra_data = SO3_GROUP.skew_matrix_from_vector(data)[:, 1:, 0]
horizontal_data = data
horizontal_data[:, 0] = 0
horizontal_mean = METRIC.mean(horizontal_data)
projected_horizontal_data = SO3_GROUP.matrix_from_rotation_vector(horizontal_data).dot(np.array([1, 0, 0]))
projected_horizontal_mean = SO3_GROUP.matrix_from_rotation_vector(horizontal_mean).dot(np.array([1, 0, 0]))
assert S2_SPHERE.belongs(projected_horizontal_mean).all()
print(Sphere_Metric.dist(frechet_mean, projected_horizontal_mean))

# change representation
vertical_random = embed(SO2_GROUP.random_uniform(n_samples=1))
shifted_data = np.matmul((SO3_GROUP.matrix_from_rotation_vector(data)), vertical_random)
projected_shifted_data = shifted_data.dot(np.array([1, 0, 0]))
Sphere_Metric.dist(projected_data, projected_shifted_data)

fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(111, projection="3d")

visualization.plot(projected_mean, ax, space='S2', color='darkgreen', s=10)
visualization.plot(frechet_mean, ax, space='S2', color='lightgreen', s=10)
visualization.plot(projected_data, ax, space='S2', color='black', alpha=0.7)

plt.show()


