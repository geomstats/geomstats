"""
Helper functions for unit tests.
"""

import unittest

import geomstats.backend as gs


class TestGeomstatsMethods(unittest.TestCase):
    def assertScalar(self, result, n_samples=1, depth=1):
        return self.assertTrue(
            gs.allclose(result.shape, (n_samples, depth, 1)),
            '\nresult.shape = {} instead of {}'.format(
                result.shape, (n_samples, depth, 1)))

    def assertVector(self, result, n_samples=1, depth=1, dim=1):
        return self.assertTrue(
            gs.allclose(result.shape, (n_samples, depth, dim)),
            '\nresult.shape = {} instead of {}'.format(
                result.shape, (n_samples, depth, dim)))

    def assertAllClose(self, result, expected):
        result = gs.asarray(result)
        expected = gs.asarray(expected)
        return self.assertTrue(
            gs.allclose(result, expected),
            '\nresult.shape = {}'
            '\nexpected.shape = {}'
            '\nresult = {}'
            '\nexpected = {}'.format(
                result.shape,
                expected.shape,
                result,
                expected))

    def check_shape_belongs(self, space):
        point = space.random_uniform()
        belongs = space.belongs(point)

        self.assertScalar(belongs)

    def check_shape_belongs_vectorization(self, space, n_samples):
        n_samples = self.n_samples

        points = self.space.random_uniform(n_samples=n_samples)
        belongs = self.space.belongs(points)

        self.assertScalar(belongs, n_samples)

    def check_shape_belongs_vectorization_with_depth(self,
                                                     space,
                                                     n_samples,
                                                     depth):
        n_samples = self.n_samples
        depth = self.depth

        points = self.space.random_uniform(n_samples=n_samples, depth=depth)
        belongs = self.space.belongs(points)

        self.assertScalar(belongs, n_samples, depth)

    def check_shape_random_uniform(self, space, vec_dim):
        point = space.random_uniform()
        self.assertVector(point, dim=vec_dim)

    def check_shape_random_uniform_vectorization(self,
                                                 space,
                                                 n_samples,
                                                 vec_dim):
        point = space.random_uniform(n_samples)
        self.assertVector(point, n_samples=n_samples, dim=vec_dim)

    def check_shape_random_uniform_vectorization_with_depth(self,
                                                            space,
                                                            n_samples,
                                                            depth,
                                                            vec_dim):
        point = space.random_uniform(n_samples, depth)
        self.assertVector(point, n_samples=n_samples, depth=depth, dim=vec_dim)

    def assert_random_uniform_and_belongs(self, space):
        point = self.space.random_uniform()

        self.assertTrue(self.space.belongs(point))

    def assert_random_uniform_and_belongs_vectorization(self,
                                                        space,
                                                        n_samples):
        n_samples = self.n_samples

        points = self.space.random_uniform(n_samples=n_samples)

        self.assertTrue(gs.all(self.space.belongs(points)))

    def assert_random_uniform_and_belongs_vectorization_with_depth(self,
                                                                   space,
                                                                   n_samples,
                                                                   depth):
        n_samples = self.n_samples
        depth = self.depth

        points = self.space.random_uniform(n_samples=n_samples, depth=depth)
        belongs = self.space.belongs(points)

        self.assertTrue(gs.all(belongs))

    def check_shape_exp_vectorization(self, space, n_samples, dim):
        depth = 1

        one_vec = self.space.random_uniform()
        one_base_point = self.space.random_uniform()
        n_vecs = self.space.random_uniform(n_samples=n_samples)
        n_base_points = self.space.random_uniform(n_samples=n_samples)

        one_tangent_vec = self.space.projection_to_tangent_space(
            one_vec, base_point=one_base_point)
        result = self.metric.exp(one_tangent_vec, one_base_point)
        self.assertVector(result, 1, depth, dim)

        n_tangent_vecs = self.space.projection_to_tangent_space(
            n_vecs, base_point=one_base_point)
        result = self.metric.exp(n_tangent_vecs, one_base_point)
        self.assertVector(result, n_samples, depth, dim)

        expected = gs.zeros((n_samples, dim))
        for i in range(n_samples):
            expected[i] = self.metric.exp(n_tangent_vecs[i], one_base_point)
        expected = to_vector(expected)
        self.assertAllClose(result, expected)

        one_tangent_vec = self.space.projection_to_tangent_space(
            one_vec, base_point=n_base_points)
        result = self.metric.exp(one_tangent_vec, n_base_points)
        self.assertVector(result, n_samples, depth, dim)

        expected = gs.zeros((n_samples, dim))
        for i in range(n_samples):
            expected[i] = self.metric.exp(one_tangent_vec[i], n_base_points[i])
        expected = to_vector(expected)
        self.assertAllClose(result, expected)

        n_tangent_vecs = self.space.projection_to_tangent_space(
            n_vecs, base_point=n_base_points)
        result = self.metric.exp(n_tangent_vecs, n_base_points)
        self.assertVector(result, n_samples, depth, dim)

        expected = gs.zeros((n_samples, dim))
        for i in range(n_samples):
            expected[i] = self.metric.exp(n_tangent_vecs[i], n_base_points[i])
        expected = to_vector(expected)
        self.assertAllClose(result, expected)

    def check_shape_exp_vectorization_with_depth(self,
                                                 space,
                                                 n_samples,
                                                 depth,
                                                 dim):
        one_vec = self.space.random_uniform(
            n_samples=1, depth=depth)
        one_base_point = self.space.random_uniform(
            n_samples=1, depth=depth)
        n_vecs = self.space.random_uniform(
            n_samples=n_samples, depth=depth)
        n_base_points = self.space.random_uniform(
            n_samples=n_samples, depth=depth)

        one_tangent_vec = self.space.projection_to_tangent_space(
            one_vec, base_point=one_base_point)
        result = self.metric.exp(one_tangent_vec, one_base_point)
        self.assertVector(result, 1, depth, dim)
        expected = gs.zeros((1, depth, dim))
        for j in range(depth):
            expected[0, j] = self.metric.exp(
                one_tangent_vec[0, j], one_base_point[0, j])
        expected = to_vector(expected)
        self.assertAllClose(result, expected)

        n_tangent_vecs = self.space.projection_to_tangent_space(
            n_vecs, base_point=one_base_point)
        result = self.metric.exp(n_tangent_vecs, one_base_point)
        self.assertVector(result, n_samples, depth, dim)
        expected = gs.zeros((n_samples, depth, dim))
        for i in range(n_samples):
            for j in range(depth):
                expected[i, j] = self.metric.exp(
                    n_tangent_vecs[i, j], one_base_point[0, j])
        expected = to_vector(expected)
        self.assertAllClose(result, expected)

        one_tangent_vec = self.space.projection_to_tangent_space(
            one_vec, base_point=n_base_points)
        result = self.metric.exp(one_tangent_vec, n_base_points)
        self.assertVector(result, n_samples, depth, dim)

        expected = gs.zeros((n_samples, depth, dim))
        for i in range(n_samples):
            for j in range(depth):
                expected[i, j] = self.metric.exp(
                    one_tangent_vec[i, j], n_base_points[i, j])
        expected = to_vector(expected)
        self.assertAllClose(result, expected)

        n_tangent_vecs = self.space.projection_to_tangent_space(
            n_vecs, base_point=n_base_points)
        result = self.metric.exp(n_tangent_vecs, n_base_points)
        self.assertVector(result, n_samples, depth, dim)
        expected = gs.zeros((n_samples, depth, dim))
        for i in range(n_samples):
            for j in range(depth):
                expected[i, j] = self.metric.exp(
                    n_tangent_vecs[i, j], n_base_points[i, j])
        expected = to_vector(expected)
        self.assertAllClose(result, expected)

    def check_shape_log_vectorization(self, space, n_samples, dim):
        depth = 1

        one_point = self.space.random_uniform()
        one_base_point = self.space.random_uniform()
        n_points = self.space.random_uniform(n_samples=n_samples)
        n_base_points = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.log(one_point, one_base_point)
        self.assertVector(result, 1, depth, dim)

        result = self.metric.log(n_points, one_base_point)
        self.assertVector(result, n_samples, depth, dim)

        result = self.metric.log(one_point, n_base_points)
        self.assertVector(result, n_samples, depth, dim)

        result = self.metric.log(n_points, n_base_points)
        self.assertVector(result, n_samples, depth, dim)

    def check_shape_log_vectorization_with_depth(self,
                                                 space,
                                                 n_samples,
                                                 depth,
                                                 dim):
        one_point = self.space.random_uniform(
            n_samples=1, depth=depth)
        one_base_point = self.space.random_uniform(
            n_samples=1, depth=depth)
        n_points = self.space.random_uniform(
            n_samples=n_samples, depth=depth)
        n_base_points = self.space.random_uniform(
            n_samples=n_samples, depth=depth)

        result = self.metric.log(one_point, one_base_point)
        self.assertVector(result, 1, depth, dim)
        expected = gs.zeros((1, depth, dim))
        for j in range(depth):
            expected[0, j] = self.metric.log(
                one_point[0, j], one_base_point[0, j])
        expected = to_vector(expected)
        self.assertAllClose(result, expected)

        result = self.metric.log(n_points, one_base_point)
        self.assertVector(result, n_samples, depth, dim)
        expected = gs.zeros((n_samples, depth, dim))
        for i in range(n_samples):
            for j in range(depth):
                expected[i, j] = self.metric.log(
                    n_points[i, j], one_base_point[0, j])
        expected = to_vector(expected)
        self.assertAllClose(result, expected)

        result = self.metric.log(one_point, n_base_points)
        self.assertVector(result, n_samples, depth, dim)
        expected = gs.zeros((n_samples, depth, dim))
        for i in range(n_samples):
            for j in range(depth):
                expected[i, j] = self.metric.log(
                    one_point[0, j], n_base_points[i, j])
        expected = to_vector(expected)
        self.assertAllClose(result, expected)

        result = self.metric.log(n_points, n_base_points)
        self.assertVector(result, n_samples, depth, dim)
        expected = gs.zeros((n_samples, depth, dim))
        for i in range(n_samples):
            for j in range(depth):
                expected[i, j] = self.metric.log(
                    n_points[i, j], n_base_points[i, j])
        expected = to_vector(expected)
        self.assertAllClose(result, expected)

    def check_shape_squared_dist_vectorization(self, space, metric, n_samples):
        depth = 1

        one_point_a = self.space.random_uniform()
        one_point_b = self.space.random_uniform()
        n_points_a = self.space.random_uniform(n_samples=n_samples)
        n_points_b = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.squared_dist(one_point_a, one_point_b)
        self.assertScalar(result, 1, depth)

        result = self.metric.squared_dist(n_points_a, one_point_b)
        self.assertScalar(result, n_samples, depth)

        result = self.metric.squared_dist(one_point_a, n_points_b)
        self.assertScalar(result, n_samples, depth)

        result = self.metric.squared_dist(n_points_a, n_points_b)
        self.assertScalar(result, n_samples, depth)

    def check_shape_squared_dist_vectorization_with_depth(self,
                                                          space,
                                                          metric,
                                                          n_samples,
                                                          depth):
        one_point_a = self.space.random_uniform(
            n_samples=1, depth=depth)
        one_point_b = self.space.random_uniform(
            n_samples=1, depth=depth)
        n_points_a = self.space.random_uniform(
            n_samples=n_samples, depth=depth)
        n_points_b = self.space.random_uniform(
            n_samples=n_samples, depth=depth)

        result = self.metric.squared_dist(one_point_a, one_point_b)
        self.assertScalar(result, 1, depth)

        result = self.metric.squared_dist(n_points_a, one_point_b)
        self.assertScalar(result, n_samples, depth)

        result = self.metric.squared_dist(one_point_a, n_points_b)
        self.assertScalar(result, n_samples, depth)

        result = self.metric.squared_dist(n_points_a, n_points_b)
        self.assertScalar(result, n_samples, depth)

    def check_shape_dist_vectorization(self, space, metric, n_samples):
        depth = 1

        one_point_a = self.space.random_uniform()
        one_point_b = self.space.random_uniform()
        n_points_a = self.space.random_uniform(n_samples=n_samples)
        n_points_b = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.dist(one_point_a, one_point_b)
        self.assertScalar(result, 1, depth)

        result = self.metric.dist(n_points_a, one_point_b)
        self.assertScalar(result, n_samples, depth)

        result = self.metric.dist(one_point_a, n_points_b)
        self.assertScalar(result, n_samples, depth)

        result = self.metric.dist(n_points_a, n_points_b)
        self.assertScalar(result, n_samples, depth)

    def check_shape_dist_vectorization_with_depth(self,
                                                  space,
                                                  metric,
                                                  n_samples,
                                                  depth):
        one_point_a = self.space.random_uniform(
            n_samples=1, depth=depth)
        one_point_b = self.space.random_uniform(
            n_samples=1, depth=depth)
        n_points_a = self.space.random_uniform(
            n_samples=n_samples, depth=depth)
        n_points_b = self.space.random_uniform(
            n_samples=n_samples, depth=depth)

        result = self.metric.dist(one_point_a, one_point_b)
        self.assertScalar(result, 1, depth)

        result = self.metric.dist(n_points_a, one_point_b)
        self.assertScalar(result, n_samples, depth)

        result = self.metric.dist(one_point_a, n_points_b)
        self.assertScalar(result, n_samples, depth)

        result = self.metric.dist(n_points_a, n_points_b)
        self.assertScalar(result, n_samples, depth)


def to_scalar(expected):
    expected = gs.to_ndarray(expected, to_ndim=1)
    expected = gs.to_ndarray(expected, to_ndim=2, axis=-1)
    expected = gs.to_ndarray(expected, to_ndim=3, axis=-1)
    return expected


def to_vector(expected):
    expected = gs.to_ndarray(expected, to_ndim=2)
    expected = gs.to_ndarray(expected, to_ndim=3, axis=1)
    return expected


def to_matrix(expected):
    expected = gs.to_ndarray(expected, to_ndim=3)
    expected = gs.to_ndarray(expected, to_ndim=4, axis=1)
    return expected


def left_log_then_exp_from_identity(metric, point):
    aux = metric.left_log_from_identity(point=point)
    result = metric.left_exp_from_identity(tangent_vec=aux)
    return result


def left_exp_then_log_from_identity(metric, tangent_vec):
    aux = metric.left_exp_from_identity(tangent_vec=tangent_vec)
    result = metric.left_log_from_identity(point=aux)
    return result


def log_then_exp_from_identity(metric, point):
    aux = metric.log_from_identity(point=point)
    result = metric.exp_from_identity(tangent_vec=aux)
    return result


def exp_then_log_from_identity(metric, tangent_vec):
    aux = metric.exp_from_identity(tangent_vec=tangent_vec)
    result = metric.log_from_identity(point=aux)
    return result


def log_then_exp(metric, point, base_point):
    aux = metric.log(point=point,
                     base_point=base_point)
    result = metric.exp(tangent_vec=aux,
                        base_point=base_point)
    return result


def exp_then_log(metric, tangent_vec, base_point):
    aux = metric.exp(tangent_vec=tangent_vec,
                     base_point=base_point)
    result = metric.log(point=aux,
                        base_point=base_point)
    return result


def group_log_then_exp_from_identity(group, point):
    aux = group.group_log_from_identity(point=point)
    result = group.group_exp_from_identity(tangent_vec=aux)
    return result


def group_exp_then_log_from_identity(group, tangent_vec):
    aux = group.group_exp_from_identity(tangent_vec=tangent_vec)
    result = group.group_log_from_identity(point=aux)
    return result


def group_log_then_exp(group, point, base_point):
    aux = group.group_log(point=point,
                          base_point=base_point)
    result = group.group_exp(tangent_vec=aux,
                             base_point=base_point)
    return result


def group_exp_then_log(group, tangent_vec, base_point):
    aux = group.group_exp(tangent_vec=tangent_vec,
                          base_point=base_point)
    result = group.group_log(point=aux,
                             base_point=base_point)
    return result
