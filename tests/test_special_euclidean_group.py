"""
Unit tests for special euclidean group SE(n).

Note: Only the *canonical* left- and right- invariant
metrics on SE(3) are tested here. Other invariant
metrics are tested with the tests of the invariant_metric
module.
"""

import unittest

import geomstats.backend as gs
import tests.helper as helper

from geomstats.invariant_metric import InvariantMetric
from geomstats.spd_matrices_space import SPDMatricesSpace
from geomstats.special_euclidean_group import SpecialEuclideanGroup

# Tolerance for errors on predicted vectors, relative to the *norm*
# of the vector, as opposed to the standard behavior of gs.allclose
# where it is relative to each element of the array
RTOL = 1e-5

# TODO(nina): factorize tests vectorization se3 and so3
# TODO(nina): check docstrings
# TODO(nina): add local functions to decrease the number of for loops


class TestSpecialEuclideanGroupMethods(unittest.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        n = 3
        group = SpecialEuclideanGroup(n=n)
        spd_matrices_space = SPDMatricesSpace(n=group.dimension)

        # Points

        # -- Rotation vectors with angles
        # 0, close to 0, closely lower than pi, pi,
        # between pi and 2pi, closely larger than 2pi, 2pi,
        # and closely larger than 2pi
        with_angle_0 = gs.zeros(6)
        with_angle_close_0 = (1e-10 * gs.array([1., -1., 1., 0., 0., 0.])
                              + gs.array([0., 0., 0., 1., 5., 2]))
        with_angle_close_pi_low = ((gs.pi - 1e-9) / gs.sqrt(2)
                                   * gs.array([0., 1., -1, 0., 0., 0.])
                                   + gs.array([0., 0., 0., -100., 0., 2.]))
        with_angle_pi = (gs.pi / gs.sqrt(3)
                         * gs.array([1., 1., -1, 0., 0., 0.])
                         + gs.array([0., 0., 0., -10.2, 0., 2.6]))
        with_angle_close_pi_high = ((gs.pi + 1e-9) / gs.sqrt(3)
                                    * gs.array([-1., 1., -1, 0., 0., 0.])
                                    + gs.array([0., 0., 0., -100., 0., 2.]))
        with_angle_in_pi_2pi = ((gs.pi + 0.3) / gs.sqrt(5)
                                * gs.array([-2., 1., 0., 0., 0., 0.])
                                + gs.array([0., 0., 0., -100., 0., 2.]))
        with_angle_close_2pi_low = ((2 * gs.pi - 1e-9) / gs.sqrt(6)
                                    * gs.array([2., 1., -1, 0., 0., 0.])
                                    + gs.array([0., 0., 0., 8., 555., -2.]))
        with_angle_2pi = (2 * gs.pi / gs.sqrt(3)
                          * gs.array([1., 1., -1, 0., 0., 0.])
                          + gs.array([0., 0., 0., 1., 8., -10.]))
        with_angle_close_2pi_high = ((2 * gs.pi + 1e-9) / gs.sqrt(2)
                                     * gs.array([1., 0., -1, 0., 0., 0.])
                                     + gs.array([0., 0., 0., 1., 8., -10.]))

        point_1 = gs.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        point_2 = gs.array([0.5, 0., -0.3, 0.4, 5., 60.])

        translation_large = gs.array([0., 0., 0., 0.4, 0.5, 0.6])
        translation_small = gs.array([0., 0., 0., 0.5, 0.6, 0.7])
        rot_with_parallel_trans = gs.array([gs.pi / 3., 0., 0.,
                                           1., 0., 0.])

        elements = {'with_angle_0': with_angle_0,
                    'with_angle_close_0': with_angle_close_0,
                    'with_angle_close_pi_low': with_angle_close_pi_low,
                    'with_angle_pi': with_angle_pi,
                    'with_angle_close_pi_high': with_angle_close_pi_high,
                    'with_angle_in_pi_2pi': with_angle_in_pi_2pi,
                    'with_angle_close_2pi_low': with_angle_close_2pi_low,
                    'with_angle_2pi': with_angle_2pi,
                    'with_angle_close_2pi_high': with_angle_close_2pi_high,
                    'translation_large': translation_large,
                    'translation_small': translation_small,
                    'point_1': point_1,
                    'point_2': point_2,
                    'rot_with_parallel_trans': rot_with_parallel_trans}

        # Metrics - only diagonals
        diag_mat_at_identity = gs.zeros([group.dimension, group.dimension])
        diag_mat_at_identity[0:3, 0:3] = 2 * gs.eye(3)
        diag_mat_at_identity[3:6, 3:6] = 3 * gs.eye(3)

        left_diag_metric = InvariantMetric(
                   group=group,
                   inner_product_mat_at_identity=diag_mat_at_identity,
                   left_or_right='left')
        right_diag_metric = InvariantMetric(
                   group=group,
                   inner_product_mat_at_identity=diag_mat_at_identity,
                   left_or_right='right')

        mat_at_identity = spd_matrices_space.random_uniform()

        left_metric = InvariantMetric(
                   group=group,
                   inner_product_mat_at_identity=mat_at_identity,
                   left_or_right='left')
        right_metric = InvariantMetric(
                   group=group,
                   inner_product_mat_at_identity=mat_at_identity,
                   left_or_right='right')

        metrics = {'left_canonical': group.left_canonical_metric,
                   'right_canonical': group.right_canonical_metric,
                   'left_diag': left_diag_metric,
                   'right_diag': right_diag_metric}
        # 'left': left_metric,
        # 'right': right_metric}

        self.group = group
        self.metrics = metrics
        self.elements = elements
        self.angles_close_to_pi = ['with_angle_close_pi_low',
                                   'with_angle_pi',
                                   'with_angle_close_pi_high']
        self.n_samples = 3

    def test_random_and_belongs(self):
        """
        Test that the random uniform method samples
        on the special euclidean group.
        """
        base_point = self.group.random_uniform()
        self.assertTrue(self.group.belongs(base_point))

    def test_random_and_belongs_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        self.assertTrue(self.group.belongs(points))

    def test_regularize(self):
        point = self.elements['with_angle_0']
        result = self.group.regularize(point)
        expected = point
        expected = helper.to_vector(expected)
        gs.testing.assert_allclose(result, expected)

        less_than_pi = ['with_angle_close_0',
                        'with_angle_close_pi_low']
        for angle_type in less_than_pi:
            point = self.elements[angle_type]
            result = self.group.regularize(point)
            expected = point
            expected = helper.to_vector(expected)
            self.assertTrue(gs.allclose(result, expected), angle_type)

        # Note: by default, the rotation vector is inverted by
        # the function regularize when the angle of the rotation is pi.
        # TODO(nina): should we modify this?
        angle_type = 'with_angle_pi'
        point = self.elements[angle_type]
        result = self.group.regularize(point)

        expected = gs.zeros(6)
        expected[:3] = point[:3]
        expected[3:6] = point[3:6]
        expected = helper.to_vector(expected)

        self.assertTrue(gs.allclose(result, expected),
                        '\n{}'
                        '\npoint = {}'
                        '\nresult = {}'
                        '\nexpected = {}'.format(
                            angle_type,
                            point,
                            result,
                            expected))
        angle_type = 'with_angle_close_pi_high'
        point = self.elements[angle_type]
        result = self.group.regularize(point)
        expected = gs.zeros(6)
        expected[:3] = point[:3] / gs.linalg.norm(point[:3]) * gs.pi
        expected[3:6] = point[3:6]
        self.assertTrue(gs.allclose(result, expected), angle_type)

        in_pi_2pi = ['with_angle_in_pi_2pi',
                     'with_angle_close_2pi_low']

        for angle_type in in_pi_2pi:
            point = self.elements[angle_type]
            angle = gs.linalg.norm(point[:3])
            new_angle = gs.pi - (angle - gs.pi)

            result = self.group.regularize(point)
            expected = gs.zeros(6)
            expected[:3] = - new_angle * (point[:3] / angle)
            expected[3:6] = point[3:6]
            expected = helper.to_vector(expected)

            self.assertTrue(gs.allclose(result, expected), angle_type)

        angle_type = 'with_angle_2pi'
        point = self.elements[angle_type]
        result = self.group.regularize(point)
        expected = gs.zeros(6)
        expected[:3] = gs.array([0., 0., 0.])
        expected[3:6] = point[3:6]
        expected = helper.to_vector(expected)
        self.assertTrue(gs.allclose(result, expected), angle_type)

        angle_type = 'with_angle_close_2pi_high'
        point = self.elements[angle_type]
        angle = gs.linalg.norm(point[:3])
        new_angle = angle - 2 * gs.pi

        result = self.group.regularize(point)

        expected = gs.zeros(6)
        expected[:3] = new_angle * point[:3] / angle
        expected[3:6] = point[3:6]
        expected = helper.to_vector(expected)
        self.assertTrue(gs.allclose(result, expected),
                        '\n{}'
                        '\npoint = {}'
                        '\nresult = {}'
                        '\nexpected = {}'.format(
                                        angle_type,
                                        point,
                                        result,
                                        expected))

    def test_regularize_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        regularized_points = self.group.regularize(points)

        self.assertTrue(gs.allclose(regularized_points.shape,
                                    (n_samples, self.group.dimension)))

    def test_compose(self):
        # Composition by identity, on the right
        # Expect the original transformation
        point = self.elements['point_1']
        result = self.group.compose(point,
                                    self.group.identity)
        expected = point
        expected = helper.to_vector(expected)
        gs.testing.assert_allclose(result, expected)

        # Composition by identity, on the left
        # Expect the original transformation
        result = self.group.compose(self.group.identity,
                                    point)
        expected = point
        expected = helper.to_vector(expected)
        gs.testing.assert_allclose(result, expected)

        # Composition of translations (no rotational part)
        # Expect the sum of the translations
        result = self.group.compose(self.elements['translation_small'],
                                    self.elements['translation_large'])
        expected = (self.elements['translation_small']
                    + self.elements['translation_large'])
        expected = helper.to_vector(expected)
        gs.testing.assert_allclose(result, expected)

    def test_compose_and_inverse(self):
        point = self.elements['point_1']
        inv_point = self.group.inverse(point)
        # Compose transformation by its inverse on the right
        # Expect the group identity
        result = self.group.compose(point, inv_point)
        expected = self.group.identity
        expected = helper.to_vector(expected)
        gs.testing.assert_allclose(result, expected, atol=1e-8)

        # Compose transformation by its inverse on the left
        # Expect the group identity
        result = self.group.compose(inv_point, point)
        expected = self.group.identity
        expected = helper.to_vector(expected)
        gs.testing.assert_allclose(result, expected, atol=1e-8)

    def test_compose_vectorization(self):
        n_samples = self.n_samples
        n_points_a = self.group.random_uniform(n_samples=n_samples)
        n_points_b = self.group.random_uniform(n_samples=n_samples)
        one_point = self.group.random_uniform(n_samples=1)

        result = self.group.compose(one_point,
                                    n_points_a)
        self.assertTrue(result.shape == (n_samples, self.group.dimension))

        result = self.group.compose(n_points_a,
                                    one_point)
        self.assertTrue(result.shape == (n_samples, self.group.dimension))

        result = self.group.compose(n_points_a,
                                    n_points_b)
        self.assertTrue(result.shape == (n_samples, self.group.dimension))

    def test_inverse_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        result = self.group.inverse(points)
        self.assertTrue(result.shape == (n_samples, self.group.dimension))

    def test_left_jacobian_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        jacobians = self.group.jacobian_translation(point=points,
                                                    left_or_right='left')
        self.assertTrue(gs.allclose(
                         jacobians.shape,
                         (n_samples,
                          self.group.dimension, self.group.dimension)))

    def test_exp_from_identity_vectorization(self):
        n_samples = self.n_samples
        for metric in self.metrics.values():
            tangent_vecs = self.group.random_uniform(n_samples=n_samples)
            results = metric.exp_from_identity(tangent_vecs)

            self.assertTrue(gs.allclose(results.shape,
                                        (n_samples, self.group.dimension)))

    def test_log_from_identity_vectorization(self):
        n_samples = self.n_samples
        for metric in self.metrics.values():
            points = self.group.random_uniform(n_samples=n_samples)
            results = metric.log_from_identity(points)

            self.assertTrue(gs.allclose(results.shape,
                                        (n_samples, self.group.dimension)))

    def test_exp_vectorization(self):
        n_samples = self.n_samples

        for metric_type in self.metrics:
            metric = self.metrics[metric_type]

            one_tangent_vec = self.group.random_uniform(n_samples=1)
            one_base_point = self.group.random_uniform(n_samples=1)
            n_tangent_vec = self.group.random_uniform(n_samples=n_samples)
            n_base_point = self.group.random_uniform(n_samples=n_samples)

            # Test with the 1 base point, and n tangent vecs
            result = metric.exp(n_tangent_vec, one_base_point)
            self.assertTrue(gs.allclose(result.shape,
                                        (n_samples, self.group.dimension)))
            expected = gs.vstack([metric.exp(tangent_vec, one_base_point)
                                  for tangent_vec in n_tangent_vec])
            self.assertTrue(gs.allclose(expected.shape,
                                        (n_samples, self.group.dimension)))

            self.assertTrue(gs.allclose(expected, result),
                            'with metric {}:\n'
                            'result:\n{}\n'
                            'expected:\n{}'.format(metric_type,
                                                   result,
                                                   expected))

            # Test with the several base point, and one tangent vec
            result = metric.exp(one_tangent_vec, n_base_point)
            self.assertTrue(gs.allclose(result.shape,
                                        (n_samples, self.group.dimension)))
            expected = gs.vstack([metric.exp(one_tangent_vec, base_point)
                                  for base_point in n_base_point])
            self.assertTrue(gs.allclose(expected.shape,
                                        (n_samples, self.group.dimension)))
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

            # Test with the same number n of base point and n tangent vec
            result = metric.exp(n_tangent_vec, n_base_point)
            self.assertTrue(gs.allclose(result.shape,
                                        (n_samples, self.group.dimension)))
            expected = gs.vstack([metric.exp(tangent_vec, base_point)
                                  for tangent_vec, base_point in zip(
                                                               n_tangent_vec,
                                                               n_base_point)])
            self.assertTrue(gs.allclose(expected.shape,
                                        (n_samples, self.group.dimension)))
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

    def test_log_vectorization(self):
        n_samples = self.n_samples
        for metric_type in self.metrics:
            metric = self.metrics[metric_type]

            one_point = self.group.random_uniform(n_samples=1)
            one_base_point = self.group.random_uniform(n_samples=1)
            n_point = self.group.random_uniform(n_samples=n_samples)
            n_base_point = self.group.random_uniform(n_samples=n_samples)

            # Test with the 1 base point, and several different points
            result = metric.log(n_point, one_base_point)
            self.assertTrue(gs.allclose(result.shape,
                                        (n_samples, self.group.dimension)))
            expected = gs.vstack([metric.log(point, one_base_point)
                                  for point in n_point])

            self.assertTrue(gs.allclose(expected.shape,
                                        (n_samples, self.group.dimension)))
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

            # Test with the several base point, and 1 point
            result = metric.log(one_point, n_base_point)
            self.assertTrue(gs.allclose(result.shape,
                                        (n_samples, self.group.dimension)))
            expected = gs.vstack([metric.log(one_point, base_point)
                                  for base_point in n_base_point])

            self.assertTrue(gs.allclose(expected.shape,
                                        (n_samples, self.group.dimension)))
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

            # Test with the same number n of base point and point
            result = metric.log(n_point, n_base_point)
            self.assertTrue(gs.allclose(result.shape,
                                        (n_samples, self.group.dimension)))
            expected = gs.vstack([metric.log(point, base_point)
                                  for point, base_point in zip(n_point,
                                                               n_base_point)])
            self.assertTrue(gs.allclose(expected.shape,
                                        (n_samples, self.group.dimension)))
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

    def test_group_exp_from_identity_vectorization(self):
        n_samples = self.n_samples
        tangent_vecs = self.group.random_uniform(n_samples=n_samples)
        results = self.group.group_exp_from_identity(tangent_vecs)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

    def test_group_log_from_identity_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        results = self.group.group_log_from_identity(points)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

    def test_group_exp_vectorization(self):
        n_samples = self.n_samples
        # Test with the 1 base_point, and several different tangent_vecs
        tangent_vecs = self.group.random_uniform(n_samples=n_samples)
        base_point = self.group.random_uniform(n_samples=1)
        results = self.group.group_exp(tangent_vecs, base_point)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

        # Test with the same number of base_points and tangent_vecs
        tangent_vecs = self.group.random_uniform(n_samples=n_samples)
        base_points = self.group.random_uniform(n_samples=n_samples)
        results = self.group.group_exp(tangent_vecs, base_points)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

        # Test with the several base_points, and 1 tangent_vec
        tangent_vec = self.group.random_uniform(n_samples=1)
        base_points = self.group.random_uniform(n_samples=n_samples)
        results = self.group.group_exp(tangent_vec, base_points)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

    def test_group_log_vectorization(self):
        n_samples = self.n_samples
        # Test with the 1 base point, and several different points
        points = self.group.random_uniform(n_samples=n_samples)
        base_point = self.group.random_uniform(n_samples=1)
        results = self.group.group_log(points, base_point)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

        # Test with the same number of base points and points
        points = self.group.random_uniform(n_samples=n_samples)
        base_points = self.group.random_uniform(n_samples=n_samples)
        results = self.group.group_log(points, base_points)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

        # Test with the several base points, and 1 point
        point = self.group.random_uniform(n_samples=1)
        base_points = self.group.random_uniform(n_samples=n_samples)
        results = self.group.group_log(point, base_points)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

    def test_group_exp_from_identity(self):
        # Group exponential of a translation (no rotational part)
        # Expect the original translation
        tangent_vec = self.elements['translation_small']
        result = self.group.group_exp(base_point=self.group.identity,
                                      tangent_vec=tangent_vec)
        expected = tangent_vec
        expected = helper.to_vector(expected)
        gs.testing.assert_allclose(result, expected)

        # Group exponential of a transformation
        # where translation is parallel to rotation axis
        # Expect the original transformation
        tangent_vec = self.elements['rot_with_parallel_trans']
        result = self.group.group_exp(
                                  base_point=self.group.identity,
                                  tangent_vec=tangent_vec)
        expected = tangent_vec
        expected = helper.to_vector(expected)
        gs.testing.assert_allclose(result, expected)

    def test_group_log_from_identity(self):
        # Group logarithm of a translation (no rotational part)
        # Expect the original translation
        point = self.elements['translation_small']
        result = self.group.group_log(base_point=self.group.identity,
                                      point=point)
        expected = point
        expected = helper.to_vector(expected)
        self.assertTrue(gs.allclose(expected, result))

        # Group logarithm of a transformation
        # where translation is parallel to rotation axis
        # Expect the original transformation
        point = self.elements['rot_with_parallel_trans']
        result = self.group.group_log(base_point=self.group.identity,
                                      point=point)
        expected = point
        expected = helper.to_vector(expected)
        self.assertTrue(gs.allclose(expected, result))

    def test_group_log_then_exp_from_identity(self):
        """
        Test that the group exponential from the identity
        and the group logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        for element_type in self.elements:
            if element_type in self.angles_close_to_pi:
                continue
            point = self.elements[element_type]
            result = helper.group_log_then_exp_from_identity(
                                                 group=self.group,
                                                 point=point)
            expected = self.group.regularize(point)
            expected = helper.to_vector(expected)
            gs.testing.assert_allclose(result, expected, atol=1e-8)

    def test_group_log_then_exp_from_identity_with_angles_close_to_pi(self):
        """
        Test that the group exponential from the identity
        and the group logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        for element_type in angle_types:
            point = self.elements[element_type]
            result = helper.group_log_then_exp_from_identity(
                                                 group=self.group,
                                                 point=point)
            expected = self.group.regularize(point)
            expected = helper.to_vector(expected)

            inv_rot_expected = gs.zeros_like(expected)
            inv_rot_expected[:, :3] = - expected[:, :3]
            inv_rot_expected[:, 3:6] = expected[:, 3:6]
            inv_rot_expected = helper.to_vector(inv_rot_expected)

            self.assertTrue(gs.allclose(result, expected)
                            or gs.allclose(result, inv_rot_expected),
                            '\n{}'
                            '\npoint = {}'
                            '\nresult = {}'
                            '\nexpected = {}'
                            '\nexpected with opp rotation = {}'.format(
                               element_type,
                               point,
                               result,
                               expected,
                               inv_rot_expected))

    def test_group_exp_then_log_from_identity(self):
        """
        Test that the group exponential from the identity
        and the group logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        for element_type in self.elements:
            if element_type in self.angles_close_to_pi:
                continue
            tangent_vec = self.elements[element_type]
            result = helper.group_exp_then_log_from_identity(
                                                group=self.group,
                                                tangent_vec=tangent_vec)
            expected = self.group.regularize(tangent_vec)
            expected = helper.to_vector(expected)
            self.assertTrue(gs.allclose(result, expected),
                            '\n {}'
                            '\ntangent_vec = {} -> {}'
                            '\nresult = {} -> {}'
                            '\nexpected = {} -> {}'.format(
                               element_type,
                               tangent_vec, self.group.regularize(tangent_vec),
                               result, self.group.regularize(result),
                               expected, self.group.regularize(expected),))

    def test_group_exp_then_log_from_identity_with_angles_close_to_pi(self):
        """
        Test that the group exponential from the identity
        and the group logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        for element_type in angle_types:
            tangent_vec = self.elements[element_type]
            result = helper.group_exp_then_log_from_identity(
                                                group=self.group,
                                                tangent_vec=tangent_vec)
            expected = self.group.regularize(tangent_vec)
            expected = helper.to_vector(expected)

            inv_rot_expected = gs.zeros_like(expected)
            inv_rot_expected[:, :3] = - expected[:, :3]
            inv_rot_expected[:, 3:6] = expected[:, 3:6]
            inv_rot_expected = helper.to_vector(inv_rot_expected)

            self.assertTrue(gs.allclose(result, expected)
                            or gs.allclose(result, inv_rot_expected),
                            '\ntangent_vec = {}'
                            '\nresult = {}'
                            '\nexpected = {}'
                            '\nexpected with opp rotation = {}'.format(
                               tangent_vec,
                               result,
                               expected,
                               inv_rot_expected))

    def test_group_exp(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        # Tangent vector is a translation (no infinitesimal rotational part)
        # Expect the sum of the translation
        # with the translation of the reference point
        result = self.group.group_exp(
                           base_point=self.elements['translation_small'],
                           tangent_vec=self.elements['translation_large'])
        expected = (self.elements['translation_small']
                    + self.elements['translation_large'])
        expected = helper.to_vector(expected)
        gs.testing.assert_allclose(result, expected)

    def test_group_log(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        # Point is a translation (no rotational part)
        # Expect the difference of the translation
        # by the translation of the reference point
        result = self.group.group_log(
                             base_point=self.elements['translation_small'],
                             point=self.elements['translation_large'])
        expected = (self.elements['translation_large']
                    - self.elements['translation_small'])

        expected = helper.to_vector(expected)
        gs.testing.assert_allclose(result, expected)

    def test_group_log_then_exp(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for base_point in self.elements.values():
            for element_type in self.elements:
                if element_type in self.angles_close_to_pi:
                    continue
                point = self.elements[element_type]

                result = helper.group_log_then_exp(group=self.group,
                                                   point=point,
                                                   base_point=base_point)
                expected = self.group.regularize(point)
                expected = helper.to_vector(expected)
                self.assertTrue(gs.allclose(result, expected),
                                '\n{}'
                                '\npoint = {}'
                                '\nresult = {}'
                                '\nexpected = {}'.format(
                               element_type,
                               point,
                               result,
                               expected))

    def test_group_exp_then_log(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # TODO(nina): this test fails
        for base_point_type in self.elements:
            base_point = self.elements[base_point_type]
            for element_type in self.elements:
                if element_type in self.angles_close_to_pi:
                    continue
                tangent_vec = self.elements[element_type]

                result = helper.group_exp_then_log(group=self.group,
                                                   tangent_vec=tangent_vec,
                                                   base_point=base_point)
                metric = self.metrics['left_canonical']
                expected = self.group.regularize_tangent_vec(
                                                   tangent_vec=tangent_vec,
                                                   base_point=base_point,
                                                   metric=metric)
                expected = helper.to_vector(expected)

    def test_exp_from_identity_left(self):
        # Riemannian left-invariant metric given by
        # the canonical inner product on the lie algebra
        # Expect the identity function
        # because we use the riemannian left logarithm with canonical
        # inner product to parameterize the transformations
        metric = self.metrics['left_canonical']
        # General case
        tangent_rot_vec = gs.array([1., 1., 1.])  # NB: Regularized
        tangent_translation = gs.array([1., 0., -3.])
        tangent_vec = gs.concatenate([tangent_rot_vec,
                                      tangent_translation])
        result = metric.exp_from_identity(tangent_vec)
        expected = tangent_vec
        expected = helper.to_vector(expected)

        self.assertTrue(gs.allclose(result, expected))

    def test_log_from_identity_left(self):
        # Riemannian left-invariant metric given by
        # the canonical inner product on the lie algebra
        # Expect the identity function
        # because we use the riemannian left logarithm with canonical
        # inner product to parameterize the transformations

        metric = self.metrics['left_canonical']
        # General case
        rot_vec = gs.array([0.1, 1, 0.9])  # NB: Regularized
        translation = gs.array([1, -19, -3])
        transfo = gs.concatenate([rot_vec, translation])

        expected = transfo
        expected = helper.to_vector(expected)
        result = metric.log_from_identity(transfo)

        self.assertTrue(gs.allclose(result, expected))

        # Edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec = gs.array([1e-8, 0., 1e-9])  # NB: Regularized
        translation = gs.array([10000, -5.9, -93])
        transfo = gs.concatenate([rot_vec, translation])

        expected = transfo
        expected = helper.to_vector(expected)
        result = metric.log_from_identity(transfo)

        self.assertTrue(gs.allclose(result, expected))

    def test_exp_then_log_from_identity_left(self):
        """
        Test that the riemannian left exponential from the identity
        and the riemannian left logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        # Canonical inner product on the lie algebra

        for metric in [self.metrics['left_canonical'],
                       self.metrics['left_diag']]:
            for angle_type in self.elements:
                if angle_type in self.angles_close_to_pi:
                    continue
                tangent_vec = self.elements[angle_type]
                result = helper.exp_then_log_from_identity(
                                                metric=metric,
                                                tangent_vec=tangent_vec)
                expected = self.group.regularize_tangent_vec_at_identity(
                                                tangent_vec,
                                                metric=metric)
                expected = helper.to_vector(expected)
                self.assertTrue(gs.allclose(result, expected),
                                '\ntangent_vec = {}'
                                '\nresult = {}'
                                '\nexpected = {}'.format(
                                   tangent_vec,
                                   result,
                                   expected))

    def test_exp_then_log_from_identity_left_with_angles_close_to_pi(self):
        """
        Test that the riemannian left exponential from the identity
        and the riemannian left logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        # Canonical inner product on the lie algebra
        for metric in [self.metrics['left_canonical'],
                       self.metrics['left_diag']]:
            for angle_type in angle_types:
                tangent_vec = self.elements[angle_type]
                result = helper.exp_then_log_from_identity(
                                                metric=metric,
                                                tangent_vec=tangent_vec)
                expected = self.group.regularize_tangent_vec_at_identity(
                                                tangent_vec=tangent_vec,
                                                metric=metric)
                expected = helper.to_vector(expected)
                inv_rot_expected = gs.zeros_like(expected)
                inv_rot_expected[:, :3] = - expected[:, :3]
                inv_rot_expected[:, 3:6] = expected[:, 3:6]
                inv_rot_expected = helper.to_vector(inv_rot_expected)

                self.assertTrue(gs.allclose(result, expected)
                                or gs.allclose(result, inv_rot_expected),
                                '\ntangent_vec = {}'
                                '\nresult = {}'
                                '\nexpected = {}'
                                '\nexpected with opp rotation = {}'.format(
                                   tangent_vec,
                                   result,
                                   expected,
                                   inv_rot_expected))

    def test_exp_then_log_from_identity_right(self):
        """
        Test that the riemannian right exponential from the identity
        and the riemannian right logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        # Canonical inner product on the lie algebra
        for metric in [self.metrics['right_canonical'],
                       self.metrics['right_diag']]:
            for angle_type in self.elements:
                if angle_type in self.angles_close_to_pi:
                    continue
                tangent_vec = self.elements[angle_type]
                result = helper.exp_then_log_from_identity(
                                                metric=metric,
                                                tangent_vec=tangent_vec)
                expected = self.group.regularize_tangent_vec_at_identity(
                        tangent_vec=tangent_vec,
                        metric=metric)
                expected = helper.to_vector(expected)

                self.assertTrue(gs.allclose(result, expected),
                                '\ntangent_vec = {}'
                                '\nresult = {}'
                                '\nexpected = {}'.format(
                                   tangent_vec,
                                   result,
                                   expected))

    def test_exp_then_log_from_identity_right_with_angles_close_to_pi(self):
        """
        Test that the riemannian right exponential from the identity
        and the riemannian right logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        # Canonical inner product on the lie algebra
        for metric in [self.metrics['right_canonical'],
                       self.metrics['right_diag']]:
            for angle_type in angle_types:
                tangent_vec = self.elements[angle_type]
                result = helper.exp_then_log_from_identity(
                                                metric=metric,
                                                tangent_vec=tangent_vec)
                expected = self.group.regularize_tangent_vec_at_identity(
                                                tangent_vec=tangent_vec,
                                                metric=metric)
                expected = helper.to_vector(expected)
                inv_rot_expected = gs.zeros_like(expected)
                inv_rot_expected[:, :3] = - expected[:, :3]
                inv_rot_expected[:, 3:6] = expected[:, 3:6]
                inv_rot_expected = helper.to_vector(inv_rot_expected)

                self.assertTrue(gs.allclose(result, expected)
                                or gs.allclose(result, inv_rot_expected),
                                '\ntangent_vec = {}'
                                '\nresult = {}'
                                '\nexpected = {}'
                                '\nexpected with opp rotation = {}'.format(
                                   tangent_vec,
                                   result,
                                   expected,
                                   inv_rot_expected))

    def test_exp_left(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        metric = self.metrics['left_canonical']
        rot_vec_base_point = gs.array([0., 0., 0.])
        translation_base_point = gs.array([4, -1, 10000])
        transfo_base_point = gs.concatenate([rot_vec_base_point,
                                            translation_base_point])

        # Tangent vector is a translation (no infinitesimal rotational part)
        # Expect the sum of the translation
        # with the translation of the reference point
        rot_vec = gs.array([0., 0., 0.])
        translation = gs.array([1, 0, -3])
        tangent_vec = gs.concatenate([rot_vec, translation])

        result = metric.exp(base_point=transfo_base_point,
                            tangent_vec=tangent_vec)
        expected = gs.concatenate([gs.array([0., 0., 0.]),
                                   gs.array([5, -1, 9997])])
        expected = helper.to_vector(expected)
        self.assertTrue(gs.allclose(result, expected))

    def test_log_left(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        metric = self.metrics['left_canonical']
        rot_vec_base_point = gs.array([0., 0., 0.])
        translation_base_point = gs.array([4., 0., 0.])
        transfo_base_point = gs.concatenate([rot_vec_base_point,
                                            translation_base_point])

        # Point is a translation (no rotational part)
        # Expect the difference of the translation
        # by the translation of the reference point
        rot_vec = gs.array([0., 0., 0.])
        translation = gs.array([-1., -1., -1.2])
        point = gs.concatenate([rot_vec,
                                translation])

        expected = gs.concatenate([gs.array([0., 0., 0.]),
                                   gs.array([-5., -1., -1.2])])
        expected = helper.to_vector(expected)

        result = metric.log(base_point=transfo_base_point,
                            point=point)

        self.assertTrue(gs.allclose(result, expected))

    def test_log_then_exp_left(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for metric in [self.metrics['left_canonical'],
                       self.metrics['left_diag']]:
            for base_point_type in self.elements:
                base_point = self.elements[base_point_type]
                for element_type in self.elements:
                    if element_type in self.angles_close_to_pi:
                        continue
                    point = self.elements[element_type]
                    result = helper.log_then_exp(
                                                metric=metric,
                                                point=point,
                                                base_point=base_point)

                    expected = self.group.regularize(point)
                    expected = helper.to_vector(expected)

                    self.assertTrue(gs.allclose(result, expected),
                                    '\nresult = {}'
                                    '\nexpected = {}'.format(
                                   result,
                                   expected))

    def test_log_then_exp_left_with_angles_close_to_pi(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        # Canonical inner product on the lie algebra
        for metric in [self.metrics['left_canonical'],
                       self.metrics['left_diag']]:
            for base_point in self.elements.values():
                for element_type in angle_types:
                    point = self.elements[element_type]
                    result = helper.log_then_exp(
                                                metric=metric,
                                                point=point,
                                                base_point=base_point)

                    expected = self.group.regularize(point)
                    expected = helper.to_vector(expected)

                    inv_rot_expected = gs.zeros_like(expected)
                    inv_rot_expected[:, :3] = - expected[:, :3]
                    inv_rot_expected[:, 3:6] = expected[:, 3:6]
                    inv_rot_expected = helper.to_vector(inv_rot_expected)

                    self.assertTrue(gs.allclose(result, expected)
                                    or gs.allclose(result, inv_rot_expected),
                                    '\npoint = {}'
                                    '\nresult = {}'
                                    '\nexpected = {}'
                                    '\nexpected with opp rotation = {}'.format(
                                       point,
                                       result,
                                       expected,
                                       inv_rot_expected))

    def test_exp_then_log_left(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for metric in [self.metrics['left_canonical'],
                       self.metrics['left_diag']]:
            for base_point in self.elements.values():
                for element_type in self.elements:
                    if element_type in self.angles_close_to_pi:
                        continue
                    tangent_vec = self.elements[element_type]
                    result = helper.exp_then_log(
                                                metric=metric,
                                                tangent_vec=tangent_vec,
                                                base_point=base_point)

                    expected = self.group.regularize_tangent_vec(
                                                tangent_vec=tangent_vec,
                                                base_point=base_point,
                                                metric=metric)
                    expected = helper.to_vector(expected)
                    norm = gs.linalg.norm(expected)
                    atol = RTOL
                    if norm != 0:
                        atol = RTOL * norm
                    self.assertTrue(gs.allclose(result, expected, atol=atol),
                                    '\ntangent_vec = {}'
                                    '\nresult = {}'
                                    '\nexpected = {}'.format(
                                   tangent_vec,
                                   result,
                                   expected))

    def test_exp_then_log_left_with_angles_close_to_pi(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        # Canonical inner product on the lie algebra
        for metric in [self.metrics['left_canonical'],
                       self.metrics['left_diag']]:
            for base_point in self.elements.values():
                for element_type in angle_types:
                    tangent_vec = self.elements[element_type]
                    result = helper.exp_then_log(
                                                metric=metric,
                                                tangent_vec=tangent_vec,
                                                base_point=base_point)

                    expected = self.group.regularize_tangent_vec(
                                                tangent_vec=tangent_vec,
                                                base_point=base_point,
                                                metric=metric)
                    expected = helper.to_vector(expected)

                    inv_rot_expected = gs.zeros_like(expected)
                    inv_rot_expected[:, :3] = - expected[:, :3]
                    inv_rot_expected[:, 3:6] = expected[:, 3:6]
                    inv_rot_expected = helper.to_vector(inv_rot_expected)

                    self.assertTrue(gs.allclose(result, expected)
                                    or gs.allclose(result, inv_rot_expected),
                                    '\ntangent_vec = {}'
                                    '\nresult = {}'
                                    '\nexpected = {}'
                                    '\nexpected with opp rotation = {}'.format(
                                       tangent_vec,
                                       result,
                                       expected,
                                       inv_rot_expected))

    def test_log_then_exp_right(self):
        """
        Test that the riemannian right exponential and the
        riemannian right logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for metric in [self.metrics['right_canonical'],
                       self.metrics['right_diag']]:
            for base_point in self.elements.values():
                for element_type in self.elements:
                    if element_type in self.angles_close_to_pi:
                        continue
                    point = self.elements[element_type]
                    result = helper.log_then_exp(
                                                metric=metric,
                                                point=point,
                                                base_point=base_point)

                    expected = self.group.regularize(point)
                    expected = helper.to_vector(expected)
                    norm = gs.linalg.norm(expected)
                    atol = RTOL
                    if norm != 0:
                        atol = RTOL * norm
                    self.assertTrue(gs.allclose(result, expected, atol=atol),
                                    '\npoint = {}'
                                    '\nresult = {}'
                                    '\nexpected = {}'.format(
                                   point,
                                   result,
                                   expected))

    def test_log_then_exp_right_with_angles_close_to_pi(self):
        """
        Test that the riemannian right exponential and the
        riemannian right logarithm are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        # Canonical inner product on the lie algebra
        for metric in [self.metrics['right_canonical'],
                       self.metrics['right_diag']]:
            for base_point in self.elements.values():
                for element_type in angle_types:
                    point = self.elements[element_type]
                    result = helper.log_then_exp(
                                                metric=metric,
                                                point=point,
                                                base_point=base_point)

                    expected = self.group.regularize(point)
                    expected = helper.to_vector(expected)

                    inv_rot_expected = gs.zeros_like(expected)
                    inv_rot_expected[:, :3] = - expected[:, :3]
                    inv_rot_expected[:, 3:6] = expected[:, 3:6]
                    inv_rot_expected = helper.to_vector(inv_rot_expected)
                    norm = gs.linalg.norm(expected)
                    atol = RTOL
                    if norm != 0:
                        atol = RTOL * norm

                    self.assertTrue(gs.allclose(result, expected, atol=atol)
                                    or gs.allclose(result, inv_rot_expected,
                                                   atol=atol),
                                    '\npoint = {}'
                                    '\nresult = {}'
                                    '\nexpected = {}'
                                    '\nexpected with opp rotation = {}'.format(
                                       point,
                                       result,
                                       expected,
                                       inv_rot_expected))

    def test_exp_then_log_right(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # TODO(nina): this test fails.
        for metric in [self.metrics['right_canonical'],
                       self.metrics['right_diag']]:
            for base_point_type in self.elements:
                base_point = self.elements[base_point_type]
                for element_type in self.elements:
                    if element_type in self.angles_close_to_pi:
                        continue
                    tangent_vec = self.elements[element_type]
                    result = helper.exp_then_log(
                                                metric=metric,
                                                tangent_vec=tangent_vec,
                                                base_point=base_point)

                    expected = self.group.regularize_tangent_vec(
                                                tangent_vec=tangent_vec,
                                                base_point=base_point,
                                                metric=metric)

    def test_exp_then_log_right_with_angles_close_to_pi(self):
        """
        Test that the riemannian right exponential and the
        riemannian right logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # TODO(nina): This test fails
        angle_types = self.angles_close_to_pi
        # Canonical inner product on the lie algebra
        for metric in [self.metrics['right_canonical'],
                       self.metrics['right_diag']]:
            for base_point in self.elements.values():
                for element_type in angle_types:
                    tangent_vec = self.elements[element_type]
                    result = helper.exp_then_log(
                                                metric=metric,
                                                tangent_vec=tangent_vec,
                                                base_point=base_point)

                    expected = self.group.regularize_tangent_vec(
                                                tangent_vec=tangent_vec,
                                                base_point=base_point,
                                                metric=metric)

                    inv_rot_expected = gs.zeros_like(expected)
                    inv_rot_expected[:, :3] = - expected[:, :3]
                    inv_rot_expected[:, 3:6] = expected[:, 3:6]
                    norm = gs.linalg.norm(expected)
                    atol = RTOL
                    if norm != 0:
                        atol = RTOL * norm

    def test_inner_product_at_identity_vectorization(self):
        n_samples = self.n_samples
        for metric in self.metrics.values():
            one_vector_a = self.group.random_uniform(n_samples=1)
            one_vector_b = self.group.random_uniform(n_samples=1)
            n_vector_a = self.group.random_uniform(n_samples=n_samples)
            n_vector_b = self.group.random_uniform(n_samples=n_samples)

            result = metric.inner_product(one_vector_a, n_vector_b)
            expected = gs.vstack([metric.inner_product(one_vector_a, vec_b)
                                  for vec_b in n_vector_b])
            gs.testing.assert_allclose(result, expected)

            result = metric.inner_product(n_vector_a, one_vector_b)
            expected = gs.vstack([metric.inner_product(vec_a, one_vector_b)
                                  for vec_a in n_vector_a])
            gs.testing.assert_allclose(result, expected)

            result = metric.inner_product(n_vector_a, n_vector_b)
            expected = gs.vstack([metric.inner_product(vec_a, vec_b)
                                  for vec_a, vec_b in zip(n_vector_a,
                                                          n_vector_b)])
            gs.testing.assert_allclose(result, expected)

    def test_inner_product_one_base_point_vectorization(self):
        n_samples = self.n_samples
        for metric in self.metrics.values():
            one_base_point = self.group.random_uniform(n_samples=1)

            one_vector_a = self.group.random_uniform(n_samples=1)
            one_vector_b = self.group.random_uniform(n_samples=1)
            n_vector_a = self.group.random_uniform(n_samples=n_samples)
            n_vector_b = self.group.random_uniform(n_samples=n_samples)

            result = metric.inner_product(one_vector_a, n_vector_b,
                                          one_base_point)
            expected = gs.vstack([metric.inner_product(one_vector_a, vec_b,
                                                       one_base_point)
                                  for vec_b in n_vector_b])
            gs.testing.assert_allclose(result, expected)

            result = metric.inner_product(n_vector_a, one_vector_b,
                                          one_base_point)
            expected = gs.vstack([metric.inner_product(vec_a, one_vector_b,
                                                       one_base_point)
                                  for vec_a in n_vector_a])
            gs.testing.assert_allclose(result, expected)

            result = metric.inner_product(n_vector_a, n_vector_b,
                                          one_base_point)
            expected = gs.vstack([metric.inner_product(vec_a, vec_b,
                                                       one_base_point)
                                  for vec_a, vec_b in zip(n_vector_a,
                                                          n_vector_b)])
            gs.testing.assert_allclose(result, expected)

    def test_inner_product_n_base_point_vectorization(self):
        n_samples = self.n_samples
        for metric in self.metrics.values():
            n_base_point = self.group.random_uniform(n_samples=n_samples)

            one_vector_a = self.group.random_uniform(n_samples=1)
            one_vector_b = self.group.random_uniform(n_samples=1)
            n_vector_a = self.group.random_uniform(n_samples=n_samples)
            n_vector_b = self.group.random_uniform(n_samples=n_samples)

            result = metric.inner_product(one_vector_a, n_vector_b,
                                          n_base_point)
            expected = gs.vstack([metric.inner_product(one_vector_a, vec_b,
                                                       base_point)
                                  for vec_b, base_point in zip(n_vector_b,
                                                               n_base_point)])
            gs.testing.assert_allclose(result, expected)

            result = metric.inner_product(n_vector_a, one_vector_b,
                                          n_base_point)
            expected = gs.vstack([metric.inner_product(vec_a, one_vector_b,
                                                       base_point)
                                  for vec_a, base_point in zip(n_vector_a,
                                                               n_base_point)])
            gs.testing.assert_allclose(result, expected)

            result = metric.inner_product(n_vector_a, n_vector_b,
                                          n_base_point)
            expected = gs.vstack([metric.inner_product(vec_a, vec_b,
                                                       base_point)
                                  for vec_a, vec_b, base_point in zip(
                                                                n_vector_a,
                                                                n_vector_b,
                                                                n_base_point)])
            gs.testing.assert_allclose(result, expected)

    def test_squared_dist_is_symmetric(self):
        for metric_type in self.metrics:
            metric = self.metrics[metric_type]
            for point_a in self.elements.values():
                for point_b in self.elements.values():
                    point_a = self.group.regularize(point_a)
                    point_b = self.group.regularize(point_b)

                    sq_dist_a_b = metric.squared_dist(point_a, point_b)
                    sq_dist_b_a = metric.squared_dist(point_b, point_a)

                    self.assertTrue(gs.allclose(sq_dist_a_b, sq_dist_b_a),
                                    'Squared dist a - b: {}\n'
                                    'Squared dist b - a: {}\n'
                                    ' for metric {}'.format(
                                        sq_dist_a_b,
                                        sq_dist_b_a,
                                        metric_type))

    def test_dist_is_symmetric(self):
        for metric in self.metrics.values():
            for point_a in self.elements.values():
                for point_b in self.elements.values():
                    point_a = self.group.regularize(point_a)
                    point_b = self.group.regularize(point_b)

                    dist_a_b = metric.dist(point_a, point_b)
                    dist_b_a = metric.dist(point_b, point_a)
                    self.assertTrue(gs.allclose(dist_a_b, dist_b_a))

    def test_squared_dist_vectorization(self):
        n_samples = self.n_samples
        for metric_type in self.metrics:
            metric = self.metrics[metric_type]
            point_id = self.group.identity

            one_point_1 = self.group.random_uniform(n_samples=1)
            one_point_2 = self.group.random_uniform(n_samples=1)
            one_point_1 = self.group.regularize(one_point_1)
            one_point_2 = self.group.regularize(one_point_2)

            n_point_1 = self.group.random_uniform(n_samples=n_samples)
            n_point_2 = self.group.random_uniform(n_samples=n_samples)
            n_point_1 = self.group.regularize(n_point_1)
            n_point_2 = self.group.regularize(n_point_2)

            # Identity and n points 2
            result = metric.squared_dist(point_id, n_point_2)
            gs.testing.assert_allclose(result.shape, (n_samples, 1))

            expected = gs.vstack([metric.squared_dist(point_id, point_2)
                                  for point_2 in n_point_2])
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

            # n points 1 and identity
            result = metric.squared_dist(n_point_1, point_id)

            gs.testing.assert_allclose(result.shape, (n_samples, 1))

            expected = gs.vstack([metric.squared_dist(point_1, point_id)
                                  for point_1 in n_point_1])
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

            # one point 1 and n points 2
            result = metric.squared_dist(one_point_1, n_point_2)
            gs.testing.assert_allclose(result.shape, (n_samples, 1))

            expected = gs.vstack([metric.squared_dist(one_point_1, point_2)
                                  for point_2 in n_point_2])

            # n points 1 and one point 2
            result = metric.squared_dist(n_point_1, one_point_2)
            gs.testing.assert_allclose(result.shape, (n_samples, 1))

            expected = gs.vstack([metric.squared_dist(point_1, one_point_2)
                                  for point_1 in n_point_1])
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

            # n points 1 and n points 2
            result = metric.squared_dist(n_point_1, n_point_2)
            gs.testing.assert_allclose(result.shape, (n_samples, 1))

            expected = gs.vstack([metric.squared_dist(point_1, point_2)
                                  for point_1, point_2 in zip(n_point_1,
                                                              n_point_2)])
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

    def test_dist_vectorization(self):
        n_samples = self.n_samples
        for metric_type in self.metrics:
            metric = self.metrics[metric_type]
            point_id = self.group.identity

            one_point_1 = self.group.random_uniform(n_samples=1)
            one_point_2 = self.group.random_uniform(n_samples=1)
            one_point_1 = self.group.regularize(one_point_1)
            one_point_2 = self.group.regularize(one_point_2)

            n_point_1 = self.group.random_uniform(n_samples=n_samples)
            n_point_2 = self.group.random_uniform(n_samples=n_samples)
            n_point_1 = self.group.regularize(n_point_1)
            n_point_2 = self.group.regularize(n_point_2)

            # Identity and n points 2
            result = metric.dist(point_id, n_point_2)
            gs.testing.assert_allclose(result.shape, (n_samples, 1))

            expected = gs.vstack([metric.dist(point_id, point_2)
                                  for point_2 in n_point_2])
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

            # n points 1 and identity
            result = metric.dist(n_point_1, point_id)

            gs.testing.assert_allclose(result.shape, (n_samples, 1))

            expected = gs.vstack([metric.dist(point_1, point_id)
                                  for point_1 in n_point_1])
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

            # one point 1 and n points 2
            result = metric.dist(one_point_1, n_point_2)
            gs.testing.assert_allclose(result.shape, (n_samples, 1))

            expected = gs.vstack([metric.dist(one_point_1, point_2)
                                  for point_2 in n_point_2])
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

            # n points 1 and one point 2
            result = metric.dist(n_point_1, one_point_2)
            gs.testing.assert_allclose(result.shape, (n_samples, 1))

            expected = gs.vstack([metric.dist(point_1, one_point_2)
                                  for point_1 in n_point_1])
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

            # n points 1 and n points 2
            result = metric.dist(n_point_1, n_point_2)
            gs.testing.assert_allclose(result.shape, (n_samples, 1))

            expected = gs.vstack([metric.dist(point_1, point_2)
                                  for point_1, point_2 in zip(n_point_1,
                                                              n_point_2)])
            self.assertTrue(gs.allclose(result, expected),
                            'with metric {}'.format(metric_type))

    def test_group_exponential_barycenter(self):
        # TODO(nina): this test fails, the barycenter is not accurate.
        point_1 = self.group.random_uniform()
        points = gs.vstack([point_1, point_1])
        result_1 = self.group.group_exponential_barycenter(
                                points=points)
        expected_1 = self.group.regularize(point_1)

        point_2 = self.group.random_uniform()
        points = gs.vstack([point_2, point_2])
        weights = gs.array([1., 2.])
        result_2 = self.group.group_exponential_barycenter(
                                points=points,
                                weights=weights)
        expected_2 = self.group.regularize(point_2)

        points = gs.vstack([point_1, point_2])
        weights = gs.array([1., 1.])
        result_3 = self.group.group_exponential_barycenter(
                                points=points,
                                weights=weights)

        self.assertTrue(self.group.belongs(result_3))

    def test_geodesic_and_belongs(self):
        initial_point = self.group.random_uniform()
        initial_tangent_vec = gs.array([2., 0., -1., 0., 2., 3.])
        metric = self.metrics['left_canonical']
        geodesic = metric.geodesic(initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)

        t = gs.linspace(start=0, stop=1, num=100)
        points = geodesic(t)
        self.assertTrue(gs.all(self.group.belongs(points)))

    def test_geodesic_subsample(self):
        initial_point = self.group.random_uniform()
        initial_tangent_vec = gs.array([1., 0., 2., 1., 1., 1.])
        metric = self.metrics['left_canonical']
        geodesic = metric.geodesic(initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)
        n_steps = 100
        t = gs.linspace(start=0, stop=1, num=n_steps+1)
        points = geodesic(t)

        tangent_vec_step = initial_tangent_vec / n_steps
        for i in range(n_steps+1):
            point_step = metric.exp(
                tangent_vec=i * tangent_vec_step,
                base_point=initial_point)
            assert gs.all(point_step == points[i])


if __name__ == '__main__':
        unittest.main()
