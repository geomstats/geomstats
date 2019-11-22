"""
Compares the mean of a data set of unit vectors
to the projection of the mean of rotations onto the sphere,
seen as the homogeneous space 3D/2D rotations
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/Users/nicolasguigui/gits/geomstats')

import geomstats.visualization as visualization
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.geometry.special_orthogonal_group import SpecialOrthogonalGroup

EPSILON = 1e-6


def embed(point):
    """
    embeds 2D rotations in space of 3D rotations
    :param point:
    :return:
    """
    d = np.zeros((point.shape[0], 3, 3)) + np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    d[:, 1:, 1:] = point
    return d


def lift(point):
    """
    finds a representer of point in group
    """
    lifted = np.zeros((point.shape[0], 3, 3))
    for i, p in enumerate(point):
        # b = gs.linalg.sqrtm(
        #     gs.eye(2) - gs.matmul(point[:, 1:, np.newaxis], point[:, 1:, np.newaxis].transpose(0, 2, 1)))
        b = gs.linalg.sqrtm(gs.eye(2) - gs.matmul(p[1:, np.newaxis], p[1:, np.newaxis].T))
        u = - b.dot(p[1:]) / p[0]
        # u = - np.matmul(b, point[:, 1:, np.newaxis])[:, :, 0] / point[:, 0]
        lifted[i, 1:, 1:] = b
        lifted[i, :, 0] = p
        lifted[i, 0, 1:] = u

    return lifted


def project(point):
    return point.dot(np.array([1, 0, 0]))


def mean(points, weights=None, n_max_iterations=32, epsilon=EPSILON, point_type='vector'):
    """
    Frechet mean of (weighted) points.

    Parameters
    ----------
    points: array-like, shape=[n_samples, dimension]

    weights: array-like, shape=[n_samples, 1], optional
    """

    # TODO(nina): Profile this code to study performance,
    # i.e. what to do with sq_dists_between_iterates.
    def while_loop_cond(iteration, mean, variance, sq_dist):
        result = ~gs.isclose(variance, 0.) and ~gs.less_equal(sq_dist, epsilon * variance)
        print(~gs.isclose(variance, 0.), ~gs.less_equal(sq_dist, epsilon * variance))
        return result[0, 0]

    def while_loop_body(iteration, mean, variance, sq_dist):
        tangent_mean = gs.zeros_like(mean)

        logs = SO3_GROUP.rotation_vector_from_matrix(lift(points))
        tangent_mean += gs.einsum('nk,nj->j', weights, logs)

        tangent_mean /= sum_weights

        mean_next = project(SO3_GROUP.matrix_from_rotation_vector(tangent_mean))

        sq_dist = Sphere_Metric.squared_dist(mean_next, mean)
        sq_dists_between_iterates.append(sq_dist)

        variance = Sphere_Metric.variance(points=points, weights=weights, base_point=mean_next)

        mean = mean_next
        iteration += 1
        return [iteration, mean, variance, sq_dist]

    if point_type == 'vector':
        points = gs.to_ndarray(points, to_ndim=2)
    if point_type == 'matrix':
        points = gs.to_ndarray(points, to_ndim=3)
    n_points = gs.shape(points)[0]

    if weights is None:
        weights = gs.ones((n_points, 1))

    weights = gs.array(weights)
    weights = gs.to_ndarray(weights, to_ndim=2, axis=1)

    sum_weights = gs.sum(weights)

    mean = points[0]
    if point_type == 'vector':
        mean = gs.to_ndarray(mean, to_ndim=2)
    if point_type == 'matrix':
        mean = gs.to_ndarray(mean, to_ndim=3)

    if n_points == 1:
        return mean

    sq_dists_between_iterates = []
    iteration = 0
    sq_dist = gs.array([[1.0]])
    variance = Sphere_Metric.variance(points, weights, mean)

    last_iteration, mean, variance, sq_dist = gs.while_loop(lambda i, m, v, sq: while_loop_cond(i, m, v, sq),
        lambda i, m, v, sq: while_loop_body(i, m, v, sq), loop_vars=[iteration, mean, variance, sq_dist],
        maximum_iterations=n_max_iterations)

    if last_iteration == n_max_iterations:
        print('Maximum number of iterations {} reached.'
              'The mean may be inaccurate'.format(n_max_iterations))

    mean = gs.to_ndarray(mean, to_ndim=2)
    return mean, last_iteration, variance, sq_dist


SO3_GROUP = SpecialOrthogonalGroup(n=3, point_type='vector')
SO2_GROUP = SpecialOrthogonalGroup(n=2)
S2_SPHERE = Hypersphere(dimension=2)
METRIC = SO3_GROUP.bi_invariant_metric
Sphere_Metric = HypersphereMetric(dimension=2)

N_SAMPLES = 10

data = SO3_GROUP.random_uniform(n_samples=N_SAMPLES)
matrix_data = SO3_GROUP.matrix_from_rotation_vector(data)
mean = METRIC.mean(data)
projected_mean = SO3_GROUP.matrix_from_rotation_vector(mean).dot(np.array([1, 0, 0]))
assert S2_SPHERE.belongs(projected_mean)

projected_data = matrix_data.dot(np.array([1, 0, 0]))
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
assert SO3_GROUP.belongs(vertical_random, point_type='matrix')
shifted_data = np.matmul(matrix_data, vertical_random)
projected_shifted_data = shifted_data.dot(np.array([1, 0, 0]))
Sphere_Metric.dist(projected_data, projected_shifted_data)

SO3_GROUP.default_point_type = 'matrix'
shifted_mean = METRIC.mean(shifted_data)
projected_shifted_mean = shifted_mean.dot(np.array([1, 0, 0]))
print(Sphere_Metric.dist(projected_data, projected_shifted_data))

data = S2_SPHERE.random_uniform(n_samples=N_SAMPLES)
lifted_data = lift(data)
METRIC.group.default_point_type = 'vector'
lifted_mean = METRIC.mean(SO3_GROUP.rotation_vector_from_matrix(lifted_data), epsilon=EPSILON)
projected_lifted_mean = project(SO3_GROUP.matrix_from_rotation_vector(lifted_mean))
invariant_mean, a, b, c = mean(data)
frechet_mean = Sphere_Metric.mean(data)
assert S2_SPHERE.belongs(invariant_mean)
print(Sphere_Metric.dist(frechet_mean, invariant_mean))
print(Sphere_Metric.dist(projected_lifted_mean, frechet_mean))

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection="3d")

visualization.plot(invariant_mean, ax, space='S2', color='blue', s=10)
visualization.plot(frechet_mean, ax, space='S2', color='lightgreen', s=10)
visualization.plot(projected_lifted_mean, ax, space='S2', color='red', s=10)
visualization.plot(data, ax, space='S2', color='black', alpha=0.7)

plt.show()
