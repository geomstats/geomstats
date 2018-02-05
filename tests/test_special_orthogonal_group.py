"""Unit tests for special orthogonal group module."""

import numpy as np
import unittest

from geomstats.invariant_metric import InvariantMetric
import geomstats.special_orthogonal_group as special_orthogonal_group
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

import tests.helper as helper

EPSILON = 1e-5


class TestSpecialOrthogonalGroupMethods(unittest.TestCase):
    def setUp(self):
        n = 3
        group = SpecialOrthogonalGroup(n=n)

        # -- Rotation vectors with angles
        # 0, close to 0, closely lower than pi, pi,
        # between pi and 2pi, closely larger than 2pi, 2pi,
        # and closely larger than 2pi
        with_angle_0 = np.zeros(3)
        with_angle_close_0 = 1e-10 * np.array([1., -1., 1.])
        with_angle_close_pi_low = ((np.pi - 1e-9) / np.sqrt(2)
                                   * np.array([0., 1., -1]))
        with_angle_pi = np.pi / np.sqrt(3) * np.array([1., 1., -1])
        with_angle_close_pi_high = ((np.pi + 1e-9) / np.sqrt(3)
                                    * np.array([-1., 1., -1]))
        with_angle_in_pi_2pi = ((np.pi + 0.3) / np.sqrt(5)
                                * np.array([-2., 1., 0]))
        with_angle_close_2pi_low = ((2 * np.pi - 1e-9) / np.sqrt(6)
                                    * np.array([2., 1., -1]))
        with_angle_2pi = 2 * np.pi / np.sqrt(3) * np.array([1., 1., -1])
        with_angle_close_2pi_high = ((2 * np.pi + 1e-9) / np.sqrt(2)
                                     * np.array([1., 0., -1]))

        elements = {'with_angle_0': with_angle_0,
                    'with_angle_close_0': with_angle_close_0,
                    'with_angle_close_pi_low': with_angle_close_pi_low,
                    'with_angle_pi': with_angle_pi,
                    'with_angle_close_pi_high': with_angle_close_pi_high,
                    'with_angle_in_pi_2pi': with_angle_in_pi_2pi,
                    'with_angle_close_2pi_low': with_angle_close_2pi_low,
                    'with_angle_2pi': with_angle_2pi,
                    'with_angle_close_2pi_high': with_angle_close_2pi_high}

        # -- Metrics - only diagonals for now
        canonical_metric = group.bi_invariant_metric

        diag_mat = np.diag([3., 3., 3.])
        left_diag_metric = InvariantMetric(
                   group=group,
                   inner_product_mat_at_identity=diag_mat,
                   left_or_right='left')
        right_diag_metric = InvariantMetric(
                   group=group,
                   inner_product_mat_at_identity=diag_mat,
                   left_or_right='right')

        metrics = {'canonical': canonical_metric}

        # -- Set attributes
        self.group = group
        self.elements = elements
        self.angles_close_to_pi = ['with_angle_close_pi_low',
                                   'with_angle_pi',
                                   'with_angle_close_pi_high']
        self.metrics = metrics
        self.n_random_samples = 100

    def test_closest_rotation_matrix(self):
        rot_mat = np.eye(3)
        delta = 1e-12 * np.array([[0., 0., 0.],
                                  [0., 0., 1.],
                                  [0., 1., 0.]])

        rot_mat_plus_delta = rot_mat + delta
        result = special_orthogonal_group.closest_rotation_matrix(
                                                   rot_mat_plus_delta)
        expected = rot_mat
        self.assertTrue(np.allclose(result, expected))

    def test_closest_rotation_matrix_vectorization(self):
        n_samples = self.n_random_samples
        rot_vecs = self.group.random_uniform(n_samples=n_samples)
        rot_mats = self.group.matrix_from_rotation_vector(rot_vecs)
        result = special_orthogonal_group.closest_rotation_matrix(
                                                   rot_mats)
        self.assertTrue(np.allclose(result.shape,
                                    (n_samples, self.group.n, self.group.n)))

    def test_skew_matrix_from_vector(self):
        rot_vec = np.array([1., 2., 3.])
        result = special_orthogonal_group.skew_matrix_from_vector(rot_vec)

        self.assertTrue(np.allclose(np.dot(result, rot_vec), np.zeros(3)))

    def test_skew_matrix_from_vector_vectorization(self):
        n_samples = self.n_random_samples
        rot_vecs = self.group.random_uniform(n_samples=n_samples)
        result = special_orthogonal_group.skew_matrix_from_vector(rot_vecs)

        self.assertTrue(np.allclose(result.shape,
                                    (n_samples, self.group.n, self.group.n)))

    def test_random_and_belongs(self):
        rot_vec = self.group.random_uniform()
        self.assertTrue(self.group.belongs(rot_vec))

    def test_random_and_belongs_vectorization(self):
        n_samples = self.n_random_samples
        rot_vecs = self.group.random_uniform(n_samples=n_samples)
        self.assertTrue(self.group.belongs(rot_vecs))

    def test_regularize(self):
        point = self.elements['with_angle_0']
        self.assertFalse(np.linalg.norm(point) != 0)
        result = self.group.regularize(point)
        expected = point
        self.assertTrue(np.allclose(result, expected), '! angle 0 !')

        less_than_pi = ['with_angle_close_0',
                        'with_angle_close_pi_low']
        for angle_type in less_than_pi:
            point = self.elements[angle_type]
            result = self.group.regularize(point)
            expected = point
            self.assertTrue(np.allclose(result, expected), angle_type)

        # Note: by default, the rotation vector is inverted by
        # the function regularize when the angle of the rotation is pi.
        # TODO(nina): should we modify this?
        angle_type = 'with_angle_pi'
        point = self.elements[angle_type]
        result = self.group.regularize(point)
        expected = - point
        self.assertTrue(np.allclose(result, expected), angle_type)

        in_pi_2pi = ['with_angle_close_pi_high',
                     'with_angle_in_pi_2pi',
                     'with_angle_close_2pi_low']

        for angle_type in in_pi_2pi:
            point = self.elements[angle_type]
            angle = np.linalg.norm(point)
            new_angle = np.pi - (angle - np.pi)

            result = self.group.regularize(point)
            expected = - new_angle * (point / angle)
            self.assertTrue(np.allclose(result, expected), angle_type)

        angle_type = 'with_angle_2pi'
        point = self.elements[angle_type]
        result = self.group.regularize(point)
        expected = np.array([0., 0., 0.])
        self.assertTrue(np.allclose(result, expected), angle_type)

        angle_type = 'with_angle_close_2pi_high'
        point = self.elements[angle_type]
        angle = np.linalg.norm(point)
        new_angle = angle - 2 * np.pi

        result = self.group.regularize(point)
        expected = new_angle * point / angle
        self.assertTrue(np.allclose(result, expected), angle_type)

    def test_regularize_vectorization(self):
        n_samples = self.n_random_samples
        rot_vecs = self.group.random_uniform(n_samples=n_samples)
        regularized_rot_vecs = self.group.regularize(rot_vecs)

        self.assertTrue(np.allclose(regularized_rot_vecs.shape,
                                    (n_samples, self.group.dimension)))

    def test_matrix_from_rotation_vector(self):
        rot_vec_0 = self.group.identity
        rot_mat_0 = self.group.matrix_from_rotation_vector(rot_vec_0)
        expected_rot_mat_0 = np.eye(3)
        self.assertTrue(np.allclose(rot_mat_0, expected_rot_mat_0))

        rot_vec_1 = np.array([np.pi / 3., 0., 0.])
        rot_mat_1 = self.group.matrix_from_rotation_vector(rot_vec_1)
        expected_rot_mat_1 = np.array([[1., 0., 0.],
                                       [0., 0.5, -np.sqrt(3) / 2],
                                       [0., np.sqrt(3) / 2, 0.5]])
        self.assertTrue(np.allclose(rot_mat_1, expected_rot_mat_1))

        rot_vec_3 = 1e-11 * np.array([12., 1., -81.])
        angle = np.linalg.norm(rot_vec_3)
        skew_rot_vec_3 = 1e-11 * np.array([[0., 81., 1.],
                                           [-81., 0., -12.],
                                           [-1., 12., 0.]])
        coef_1 = np.sin(angle) / angle
        coef_2 = (1 - np.cos(angle)) / (angle ** 2)
        expected_rot_mat_3 = (np.identity(3)
                              + coef_1 * skew_rot_vec_3
                              + coef_2 * np.dot(skew_rot_vec_3,
                                                skew_rot_vec_3))
        rot_mat_3 = self.group.matrix_from_rotation_vector(rot_vec_3)
        self.assertTrue(np.allclose(rot_mat_3, expected_rot_mat_3))

        rot_vec_6 = np.array([.1, 1.3, -.5])
        angle = np.linalg.norm(rot_vec_6)
        skew_rot_vec_6 = np.array([[0., .5, 1.3],
                                   [-.5, 0., -.1],
                                   [-1.3, .1, 0.]])

        coef_1 = np.sin(angle) / angle
        coef_2 = (1 - np.cos(angle)) / (angle ** 2)
        rot_mat_6 = self.group.matrix_from_rotation_vector(rot_vec_6)
        expected_rot_mat_6 = (np.identity(3)
                              + coef_1 * skew_rot_vec_6
                              + coef_2 * np.dot(skew_rot_vec_6,
                                                skew_rot_vec_6))
        self.assertTrue(np.allclose(rot_mat_6, expected_rot_mat_6))

    def test_matrix_from_rotation_vector_vectorization(self):
        n_samples = self.n_random_samples
        rot_vecs = self.group.random_uniform(n_samples=n_samples)
        rot_mats = self.group.matrix_from_rotation_vector(rot_vecs)

        self.assertTrue(np.allclose(rot_mats.shape,
                                    (n_samples, self.group.n, self.group.n)))

    def test_rotation_vector_from_matrix(self):
        rot_mat = np.array([[1., 0., 0.],
                            [0., np.cos(.12), -np.sin(.12)],
                            [0, np.sin(.12), np.cos(.12)]])
        rot_vec = self.group.rotation_vector_from_matrix(rot_mat)
        expected_rot_vec = .12 * np.array([1., 0., 0.])

        self.assertTrue(np.allclose(rot_vec, expected_rot_vec))

    def test_rotation_vector_and_rotation_matrix(self):
        """
        This tests that the composition of
        rotation_vector_from_matrix
        and
        matrix_from_rotation_vector
        is the identity.
        """
        for angle_type in self.elements:
            point = self.elements[angle_type]
            if angle_type in self.angles_close_to_pi:
                continue

            rot_mat = self.group.matrix_from_rotation_vector(point)
            result = self.group.rotation_vector_from_matrix(rot_mat)

            expected = self.group.regularize(point)

            self.assertTrue(np.allclose(result, expected),
                            'for point {}:\n'
                            'result = {};'
                            ' expected = {}.'.format(angle_type,
                                                     result,
                                                     expected))

    def test_rotation_vector_and_rotation_matrix_vectorization(self):
        n_samples = self.n_random_samples
        rot_vecs = self.group.random_uniform(n_samples=n_samples)
        rot_mats = self.group.matrix_from_rotation_vector(rot_vecs)
        results = self.group.rotation_vector_from_matrix(rot_mats)

        expected = self.group.regularize(rot_vecs)
        self.assertTrue(np.allclose(results, expected))

    def test_rotation_vector_and_rotation_matrix_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        rotation_vector_from_matrix
        and
        matrix_from_rotation_vector
        is the identity.
        """
        angle_types = self.angles_close_to_pi
        for angle_type in angle_types:
            point = self.elements[angle_type]

            rot_mat = self.group.matrix_from_rotation_vector(point)
            result = self.group.rotation_vector_from_matrix(rot_mat)

            expected = self.group.regularize(point)
            inv_expected = - expected

            self.assertTrue((np.allclose(result, expected)
                            or np.allclose(result, inv_expected)),
                            'for point {}:\n'
                            'result = {}; expected = {};'
                            'inv_expected = {} '.format(angle_type,
                                                        result,
                                                        expected,
                                                        inv_expected))

    def test_compose(self):
        for element_type in self.elements:
            point = self.elements[element_type]
            # Composition by identity, on the right
            # Expect the original transformation
            result = self.group.compose(point,
                                        self.group.identity)
            expected = self.group.regularize(point)
            if element_type not in self.angles_close_to_pi:
                self.assertTrue(np.allclose(result, expected),
                                '\n{}'
                                '\nresult: {}'
                                '\nexpected: {}'.format(element_type,
                                                        result,
                                                        expected))
            else:
                inv_expected = - expected
                self.assertTrue(np.allclose(result, expected)
                                or np.allclose(result, inv_expected))

            # Composition by identity, on the left
            # Expect the original transformation
            result = self.group.compose(self.group.identity,
                                        point)
            expected = self.group.regularize(point)

            if element_type not in self.angles_close_to_pi:
                self.assertTrue(np.allclose(result, expected))
            else:
                inv_expected = - expected
                self.assertTrue(np.allclose(result, expected)
                                or np.allclose(result, inv_expected))

    def test_compose_and_inverse(self):
        for point in self.elements.values():
            inv_point = self.group.inverse(point)
            # Compose transformation by its inverse on the right
            # Expect the group identity
            result = self.group.compose(point, inv_point)
            expected = self.group.identity
            self.assertTrue(np.allclose(result, expected))

            # Compose transformation by its inverse on the left
            # Expect the group identity
            result = self.group.compose(inv_point, point)
            expected = self.group.identity
            self.assertTrue(np.allclose(result, expected))

    def test_compose_vectorization(self):
        n_samples = self.n_random_samples
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
        n_samples = self.n_random_samples
        points = self.group.random_uniform(n_samples=n_samples)
        result = self.group.inverse(points)
        self.assertTrue(result.shape == (n_samples, self.group.dimension))

    def test_left_jacobian_through_its_determinant(self):
        for angle_type in self.elements:
            point = self.elements[angle_type]
            jacobian = self.group.jacobian_translation(point=point,
                                                       left_or_right='left')
            result = np.linalg.det(jacobian)
            point = self.group.regularize(point)
            angle = np.linalg.norm(point)
            if angle_type in ['with_angle_0',
                              'with_angle_close_0',
                              'with_angle_2pi',
                              'with_angle_close_2pi_high']:
                expected = 1. + angle ** 2 / 12. + angle ** 4 / 240.
            else:
                expected = angle ** 2 / (4 * np.sin(angle / 2) ** 2)

            self.assertTrue(np.allclose(result, expected),
                            'for point {}:\n'
                            'result = {}; expected = {}.'.format(
                                                     angle_type,
                                                     result,
                                                     expected))

    def test_left_jacobian_vectorization(self):
        n_samples = self.n_random_samples
        points = self.group.random_uniform(n_samples=n_samples)
        jacobians = self.group.jacobian_translation(point=points,
                                                    left_or_right='left')
        self.assertTrue(np.allclose(
                         jacobians.shape,
                         (n_samples,
                          self.group.dimension, self.group.dimension)))

    def test_exp(self):
        """
        The Riemannian exp and log are inverse functions of each other.
        This test is the inverse of test_log's.
        """
        metric = self.metrics['canonical']
        theta = np.pi / 5
        rot_vec_base_point = theta / np.sqrt(3.) * np.array([1., 1., 1.])
        # Note: the rotation vector for the reference point
        # needs to be regularized.

        # 1: Exponential of 0 gives the reference point
        rot_vec_1 = np.array([0, 0, 0])
        expected_1 = rot_vec_base_point

        exp_1 = metric.exp(base_point=rot_vec_base_point,
                           tangent_vec=rot_vec_1)
        self.assertTrue(np.allclose(exp_1, expected_1))

        # 2: General case - computed manually
        rot_vec_2 = np.pi / 4 * np.array([1, 0, 0])
        phi = (np.pi / 10) / (np.tan(np.pi / 10))
        skew = np.array([[0., -1., 1.],
                         [1., 0., -1.],
                         [-1., 1., 0.]])
        jacobian = (phi * np.eye(3)
                    + (1 - phi) / 3 * np.ones([3, 3])
                    + np.pi / (10 * np.sqrt(3)) * skew)
        inv_jacobian = np.linalg.inv(jacobian)
        expected_2 = self.group.compose(rot_vec_base_point,
                                        np.dot(inv_jacobian, rot_vec_2))

        exp_2 = metric.exp(base_point=rot_vec_base_point,
                           tangent_vec=rot_vec_2)
        self.assertTrue(np.allclose(exp_2, expected_2))

    def test_exp_vectorization(self):
        n_samples = self.n_random_samples
        metric = self.metrics['canonical']

        # Test with the 1 base_point, and several different tangent_vecs
        tangent_vecs = self.group.random_uniform(n_samples=n_samples)
        base_point = self.group.random_uniform(n_samples=1)
        results = metric.exp(tangent_vecs, base_point)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

        # Test with the same number of base_points and tangent_vecs
        tangent_vecs = self.group.random_uniform(n_samples=n_samples)
        base_points = self.group.random_uniform(n_samples=n_samples)
        results = metric.exp(tangent_vecs, base_points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

        # Test with the several base_points, and 1 tangent_vec
        tangent_vec = self.group.random_uniform(n_samples=1)
        base_points = self.group.random_uniform(n_samples=n_samples)
        results = metric.exp(tangent_vec, base_points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

    def test_log(self):
        """
        The Riemannian exp and log are inverse functions of each other.
        This test is the inverse of test_exp's.
        """
        metric = self.metrics['canonical']
        theta = np.pi / 5.
        rot_vec_base_point = theta / np.sqrt(3.) * np.array([1., 1., 1.])
        # Note: the rotation vector for the reference point
        # needs to be regularized.

        # The Logarithm of a point at itself gives 0.
        rot_vec_1 = rot_vec_base_point
        expected_1 = np.array([0, 0, 0])
        log_1 = metric.log(base_point=rot_vec_base_point,
                           point=rot_vec_1)
        self.assertTrue(np.allclose(log_1, expected_1))

        # General case: this is the inverse test of test 1 for riemannian exp
        expected_2 = np.pi / 4 * np.array([1, 0, 0])
        phi = (np.pi / 10) / (np.tan(np.pi / 10))
        skew = np.array([[0., -1., 1.],
                         [1., 0., -1.],
                         [-1., 1., 0.]])
        jacobian = (phi * np.eye(3)
                    + (1 - phi) / 3 * np.ones([3, 3])
                    + np.pi / (10 * np.sqrt(3)) * skew)
        inv_jacobian = np.linalg.inv(jacobian)
        rot_vec_2 = self.group.compose(rot_vec_base_point,
                                       np.dot(inv_jacobian, expected_2))

        log_2 = metric.log(base_point=rot_vec_base_point,
                           point=rot_vec_2)
        self.assertTrue(np.allclose(log_2, expected_2))

    def test_log_vectorization(self):
        n_samples = self.n_random_samples
        metric = self.metrics['canonical']

        # Test with the 1 base point, and several different points
        points = self.group.random_uniform(n_samples=n_samples)
        base_point = self.group.random_uniform(n_samples=1)
        results = metric.log(points, base_point)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

        # Test with the same number of base points and points
        points = self.group.random_uniform(n_samples=n_samples)
        base_points = self.group.random_uniform(n_samples=n_samples)
        results = metric.log(points, base_points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

        # Test with the several base points, and 1 point
        point = self.group.random_uniform(n_samples=1)
        base_points = self.group.random_uniform(n_samples=n_samples)
        results = metric.log(point, base_points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

    def test_exp_from_identity_vectorization(self):
        n_samples = self.n_random_samples
        metric = self.metrics['canonical']

        tangent_vecs = self.group.random_uniform(n_samples=n_samples)
        results = metric.exp_from_identity(tangent_vecs)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

    def test_log_from_identity_vectorization(self):
        n_samples = self.n_random_samples
        metric = self.metrics['canonical']

        points = self.group.random_uniform(n_samples=n_samples)
        results = metric.log_from_identity(points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

    def test_exp_then_log_from_identity(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """

        for metric_type in self.metrics:
            for angle_type in self.elements:
                if angle_type in self.angles_close_to_pi:
                    continue

                metric = self.metrics[metric_type]
                tangent_vec = self.elements[angle_type]

                result = helper.exp_then_log_from_identity(metric, tangent_vec)

                expected = self.group.regularize(tangent_vec)

                self.assertTrue(np.allclose(result, expected),
                                '\nmetric {}\n'
                                '- on tangent_vec {}: {}\n'
                                'result = {}\n'
                                'expected = {}'.format(
                                                     metric_type,
                                                     angle_type,
                                                     tangent_vec,
                                                     result,
                                                     expected))

    def test_exp_then_log_from_identity_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        angle_types = self.angles_close_to_pi

        for metric_type in self.metrics:
            for angle_type in angle_types:

                metric = self.metrics[metric_type]
                tangent_vec = self.elements[angle_type]

                result = helper.exp_then_log_from_identity(metric, tangent_vec)

                expected = self.group.regularize(tangent_vec)
                inv_expected = - expected

                self.assertTrue(np.allclose(result, expected)
                                or np.allclose(result, inv_expected),
                                '\nmetric {}\n'
                                '- on tangent_vec {}: {}\n'
                                'result = {}\n'
                                'expected = {}'.format(
                                                     metric_type,
                                                     angle_type,
                                                     tangent_vec,
                                                     result,
                                                     expected))

    def test_log_then_exp_from_identity(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """

        for metric_type in self.metrics:
            for angle_type in self.elements:
                if angle_type in self.angles_close_to_pi:
                    continue

                metric = self.metrics[metric_type]
                point = self.elements[angle_type]

                result = helper.log_then_exp_from_identity(metric, point)
                expected = self.group.regularize(point)

                self.assertTrue(np.allclose(result, expected),
                                '\nmetric {}\n'
                                '- on point {}: {}\n'
                                'result = {}\n'
                                'expected = {}'.format(
                                                         metric_type,
                                                         angle_type,
                                                         point,
                                                         result,
                                                         expected))

    def test_log_then_exp_from_identity_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        angle_types = self.angles_close_to_pi

        for metric_type in self.metrics:
            for angle_type in angle_types:

                metric = self.metrics[metric_type]
                point = self.elements[angle_type]

                result = helper.log_then_exp_from_identity(metric, point)
                expected = self.group.regularize(point)
                inv_expected = - expected
                self.assertTrue(np.allclose(result, expected)
                                or np.allclose(result, inv_expected),
                                '\nmetric {}\n'
                                '- on point {}: {}\n'
                                'result = {}\n'
                                'expected = {}'.format(
                                                         metric_type,
                                                         angle_type,
                                                         point,
                                                         result,
                                                         expected))

    def test_exp_then_log(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        # TODO(nina): absolute tolerance for infinitesimal angles?
        for metric_type in self.metrics:
            for angle_type in self.elements:
                if angle_type in self.angles_close_to_pi:
                    continue
                for angle_type_base in self.elements:

                    metric = self.metrics[metric_type]
                    tangent_vec = self.elements[angle_type]
                    base_point = self.elements[angle_type_base]

                    result = helper.exp_then_log(metric=metric,
                                                 tangent_vec=tangent_vec,
                                                 base_point=base_point)

                    expected = helper.regularize_tangent_vec(
                                                 group=self.group,
                                                 tangent_vec=tangent_vec,
                                                 base_point=base_point)

                    self.assertTrue(np.allclose(result, expected, atol=1e-6),
                                    '\nmetric {}:\n'
                                    '- on tangent_vec {}: {} -> {}\n'
                                    '- base_point {}: {} -> {}\n'
                                    'result = {} -> {}\n'
                                    'expected = {} -> {}'.format(
                             metric_type,
                             angle_type,
                             tangent_vec, self.group.regularize(tangent_vec),
                             angle_type_base,
                             base_point, self.group.regularize(base_point),
                             result, self.group.regularize(result),
                             expected, self.group.regularize(expected)))

    def test_exp_then_log_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        angle_types = self.angles_close_to_pi
        for metric_type in self.metrics:
            for angle_type in angle_types:
                for angle_type_base in self.elements:

                    metric = self.metrics[metric_type]
                    tangent_vec = self.elements[angle_type]
                    base_point = self.elements[angle_type_base]

                    result = helper.exp_then_log(metric=metric,
                                                 tangent_vec=tangent_vec,
                                                 base_point=base_point)

                    expected = helper.regularize_tangent_vec(
                                             group=self.group,
                                             tangent_vec=tangent_vec,
                                             base_point=base_point)
                    inv_expected = - expected

                    self.assertTrue((np.allclose(result, expected)
                                     or np.allclose(result, inv_expected)),
                                    '\nmetric {}:\n'
                                    '- on tangent_vec {}: {} -> {}\n'
                                    '- base_point {}: {} -> {}\n'
                                    'result = {} -> {}\n'
                                    'expected = {} -> {}'.format(
                             metric_type,
                             angle_type,
                             tangent_vec, self.group.regularize(tangent_vec),
                             angle_type_base,
                             base_point, self.group.regularize(base_point),
                             result, self.group.regularize(result),
                             expected, self.group.regularize(expected)))

    def test_log_then_exp(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """

        for metric_type in self.metrics:
            for angle_type in self.elements:
                if angle_type in self.angles_close_to_pi:
                    continue
                for angle_type_base in self.elements:
                    metric = self.metrics[metric_type]
                    point = self.elements[angle_type]
                    base_point = self.elements[angle_type_base]

                    result = helper.log_then_exp(metric=metric,
                                                 base_point=base_point,
                                                 point=point)

                    expected = self.group.regularize(point)
                    inv_expected = - expected
                    self.assertTrue((np.allclose(result, expected)
                                     or np.allclose(result, inv_expected)),
                                    '\nmetric {}:\n'
                                    '- on point {}: {} -> {}\n'
                                    '- base_point {}: {} -> {}\n'
                                    'result = {} -> {}\n'
                                    'expected = {} -> {}'.format(
                                 metric_type,
                                 angle_type,
                                 point, self.group.regularize(point),
                                 angle_type_base,
                                 base_point, self.group.regularize(base_point),
                                 result, self.group.regularize(result),
                                 expected, self.group.regularize(expected)))

    def test_log_then_exp_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        angle_types = self.angles_close_to_pi
        for metric_type in self.metrics:
            for angle_type in angle_types:
                for angle_type_base in self.elements:
                    metric = self.metrics[metric_type]
                    point = self.elements[angle_type]
                    base_point = self.elements[angle_type_base]

                    result = helper.log_then_exp(metric=metric,
                                                 base_point=base_point,
                                                 point=point)

                    expected = self.group.regularize(point)
                    inv_expected = - expected
                    self.assertTrue((np.allclose(result, expected)
                                     or np.allclose(result, inv_expected)),
                                    '\nmetric {}:\n'
                                    '- on point {}: {} -> {}\n'
                                    '- base_point {}: {} -> {}\n'
                                    'result = {} -> {}\n'
                                    'expected = {} -> {}'.format(
                                 metric_type,
                                 angle_type,
                                 point, self.group.regularize(point),
                                 angle_type_base,
                                 base_point, self.group.regularize(base_point),
                                 result, self.group.regularize(result),
                                 expected, self.group.regularize(expected)))

    def test_group_exp_from_identity_vectorization(self):
        n_samples = self.n_random_samples
        tangent_vecs = self.group.random_uniform(n_samples=n_samples)
        results = self.group.group_exp_from_identity(tangent_vecs)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

    def test_group_log_from_identity_vectorization(self):
        n_samples = self.n_random_samples
        points = self.group.random_uniform(n_samples=n_samples)
        results = self.group.group_log_from_identity(points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

    def test_group_exp_vectorization(self):
        n_samples = self.n_random_samples
        # Test with the 1 base_point, and several different tangent_vecs
        tangent_vecs = self.group.random_uniform(n_samples=n_samples)
        base_point = self.group.random_uniform(n_samples=1)
        results = self.group.group_exp(tangent_vecs, base_point)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

        # Test with the same number of base_points and tangent_vecs
        tangent_vecs = self.group.random_uniform(n_samples=n_samples)
        base_points = self.group.random_uniform(n_samples=n_samples)
        results = self.group.group_exp(tangent_vecs, base_points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

        # Test with the several base_points, and 1 tangent_vec
        tangent_vec = self.group.random_uniform(n_samples=1)
        base_points = self.group.random_uniform(n_samples=n_samples)
        results = self.group.group_exp(tangent_vec, base_points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

    def test_group_log_vectorization(self):
        n_samples = self.n_random_samples
        # Test with the 1 base point, and several different points
        points = self.group.random_uniform(n_samples=n_samples)
        base_point = self.group.random_uniform(n_samples=1)
        results = self.group.group_log(points, base_point)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

        # Test with the same number of base points and points
        points = self.group.random_uniform(n_samples=n_samples)
        base_points = self.group.random_uniform(n_samples=n_samples)
        results = self.group.group_log(points, base_points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

        # Test with the several base points, and 1 point
        point = self.group.random_uniform(n_samples=1)
        base_points = self.group.random_uniform(n_samples=n_samples)
        results = self.group.group_log(point, base_points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples, self.group.dimension)))

    def test_group_exp_then_log_from_identity(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for angle_type in self.elements:
            if angle_type in self.angles_close_to_pi:
                continue
            tangent_vec = self.elements[angle_type]
            result = helper.group_exp_then_log_from_identity(
                                         group=self.group,
                                         tangent_vec=tangent_vec)
            expected = self.group.regularize(tangent_vec)
            self.assertTrue(np.allclose(result, expected),
                            'on tangent_vec {}'.format(angle_type))

    def test_group_exp_then_log_from_identity_with_angles_close_to_pi(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        for angle_type in angle_types:
            tangent_vec = self.elements[angle_type]
            result = helper.group_exp_then_log_from_identity(
                                         group=self.group,
                                         tangent_vec=tangent_vec)
            expected = self.group.regularize(tangent_vec)
            inv_expected = - expected
            self.assertTrue(np.allclose(result, expected)
                            or np.allclose(result, inv_expected),
                            'on tangent_vec {}'.format(angle_type))

    def test_group_log_then_exp_from_identity(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for angle_type in self.elements:
            point = self.elements[angle_type]
            result = helper.group_log_then_exp_from_identity(
                                         group=self.group,
                                         point=point)
            expected = self.group.regularize(point)
            self.assertTrue(np.allclose(result, expected),
                            'on point {}'.format(angle_type))

    def test_group_log_then_exp_from_identity_with_angles_close_to_pi(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        for angle_type in angle_types:
            point = self.elements[angle_type]
            result = helper.group_log_then_exp_from_identity(
                                         group=self.group,
                                         point=point)
            expected = self.group.regularize(point)
            inv_expected = - expected
            self.assertTrue(np.allclose(result, expected)
                            or np.allclose(result, inv_expected),
                            'on point {}'.format(angle_type))

    def test_group_exp_then_log(self):
        """
        This tests that the composition of
        log and exp gives identity.

        """
        # TODO(nina): absolute tolerance for infinitesimal angles
        for angle_type in self.elements:
            if angle_type in self.angles_close_to_pi:
                continue
            for angle_type_base in self.elements:
                tangent_vec = self.elements[angle_type]
                base_point = self.elements[angle_type_base]

                result = helper.group_exp_then_log(
                                             group=self.group,
                                             tangent_vec=tangent_vec,
                                             base_point=base_point)

                expected = helper.regularize_tangent_vec(
                                             group=self.group,
                                             tangent_vec=tangent_vec,
                                             base_point=base_point)

                self.assertTrue(np.allclose(result, expected, atol=1e-6),
                                '\n- on tangent_vec {}: {} -> {}\n'
                                '- base_point {}: {} -> {}\n'
                                'result = {} -> {}\n'
                                'expected = {} -> {}'.format(
                             angle_type,
                             tangent_vec, self.group.regularize(tangent_vec),
                             angle_type_base,
                             base_point, self.group.regularize(base_point),
                             result, self.group.regularize(result),
                             expected, self.group.regularize(expected)))

    def test_group_exp_then_log_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        angle_types = self.angles_close_to_pi
        for angle_type in angle_types:
            for angle_type_base in self.elements:
                tangent_vec = self.elements[angle_type]
                base_point = self.elements[angle_type_base]

                result = helper.group_exp_then_log(
                                             group=self.group,
                                             tangent_vec=tangent_vec,
                                             base_point=base_point)

                expected = helper.regularize_tangent_vec(
                                             group=self.group,
                                             tangent_vec=tangent_vec,
                                             base_point=base_point)

                inv_expected = - expected

                self.assertTrue((np.allclose(result, expected)
                                 or np.allclose(result, inv_expected)),
                                '\n- on tangent_vec {}: {} -> {}\n'
                                '- base_point {}: {} -> {}\n'
                                'result = {} -> {}\n'
                                'expected = {} -> {}'.format(
                             angle_type,
                             tangent_vec, self.group.regularize(tangent_vec),
                             angle_type_base,
                             base_point, self.group.regularize(base_point),
                             result, self.group.regularize(result),
                             expected, self.group.regularize(expected)))

    def test_group_log_then_exp(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """

        for angle_type in self.elements:
            if angle_type in self.angles_close_to_pi:
                continue
            for angle_type_base in self.elements:
                point = self.elements[angle_type]
                base_point = self.elements[angle_type_base]

                result = helper.group_log_then_exp(
                                             group=self.group,
                                             point=point,
                                             base_point=base_point)
                expected = self.group.regularize(point)

                self.assertTrue(np.allclose(result, expected),
                                '\n- on point {}: {} -> {}\n'
                                '- base_point {}: {} -> {}\n'
                                'result = {} -> {}\n'
                                'expected = {} -> {}'.format(
                                 angle_type,
                                 point, self.group.regularize(point),
                                 angle_type_base,
                                 base_point, self.group.regularize(base_point),
                                 result, self.group.regularize(result),
                                 expected, self.group.regularize(expected)))

    def test_group_log_then_exp_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        angle_types = self.angles_close_to_pi
        for angle_type in angle_types:
            for angle_type_base in self.elements:
                point = self.elements[angle_type]
                base_point = self.elements[angle_type_base]

                result = helper.group_log_then_exp(
                                             group=self.group,
                                             point=point,
                                             base_point=base_point)
                expected = self.group.regularize(point)
                inv_expected = - expected

                self.assertTrue((np.allclose(result, expected)
                                 or np.allclose(result, inv_expected)),
                                '\n- on point {}: {} -> {}\n'
                                '- base_point {}: {} -> {}\n'
                                'result = {} -> {}\n'
                                'expected = {} -> {}'.format(
                                 angle_type,
                                 point, self.group.regularize(point),
                                 angle_type_base,
                                 base_point, self.group.regularize(base_point),
                                 result, self.group.regularize(result),
                                 expected, self.group.regularize(expected)))

    def test_group_exponential_barycenter(self):
        rot_vec_1 = self.group.random_uniform()
        points = np.vstack([rot_vec_1, rot_vec_1])
        result_1 = self.group.group_exponential_barycenter(
                                points=points)
        expected_1 = rot_vec_1
        self.assertTrue(np.allclose(result_1, expected_1))

        rot_vec_2 = self.group.random_uniform()
        points = np.vstack([rot_vec_2, rot_vec_2])
        weights = np.array([1., 2.])
        result_2 = self.group.group_exponential_barycenter(
                                points=points,
                                weights=weights)
        expected_2 = rot_vec_2
        self.assertTrue(np.allclose(result_2, expected_2))

        points = np.vstack([rot_vec_1, rot_vec_2])
        weights = np.array([1., 2.])
        result_3 = self.group.group_exponential_barycenter(
                                points=points,
                                weights=weights)

        self.assertTrue(self.group.belongs(result_3))

    def test_squared_dist_is_symmetric(self):
        for metric in self.metrics.values():
            for angle_type_1 in self.elements:
                for angle_type_2 in self.elements:
                    point_1 = self.elements[angle_type_1]
                    point_2 = self.elements[angle_type_2]
                    point_1 = self.group.regularize(point_1)
                    point_2 = self.group.regularize(point_2)

                    sq_dist_1_2 = metric.squared_dist(point_1, point_2)
                    sq_dist_2_1 = metric.squared_dist(point_2, point_1)

                    self.assertTrue(np.allclose(sq_dist_1_2, sq_dist_2_1),
                                    'for point_1 {} and point_2 {}:\n'
                                    'squared dist from 1 to 2: {}\n'
                                    'squared dist from 2 to 1: {}\n'.format(
                                                 angle_type_1,
                                                 angle_type_2,
                                                 sq_dist_1_2,
                                                 sq_dist_2_1))

    def test_squared_dist_is_less_than_squared_pi(self):
        for metric in self.metrics.values():
            for angle_type_1 in self.elements:
                for angle_type_2 in self.elements:
                    point_1 = self.elements[angle_type_1]
                    point_2 = self.elements[angle_type_2]
                    point_1 = self.group.regularize(point_1)
                    point_2 = self.group.regularize(point_2)

                    sq_dist = metric.squared_dist(point_1, point_2)
                    diff = sq_dist - np.pi ** 2
                    self.assertTrue(diff <= 0 or abs(diff) < EPSILON,
                                    'sq_dist = {}'.format(sq_dist))

    def test_squared_dist_vectorization(self):
        n_samples = self.n_random_samples
        metric = self.metrics['canonical']
        point_1 = self.group.random_uniform(n_samples=n_samples)
        point_2 = self.group.random_uniform(n_samples=n_samples)
        point_1 = self.group.regularize(point_1)
        point_2 = self.group.regularize(point_2)

        sq_dist_1_2 = metric.squared_dist(point_1, point_2)

        self.assertTrue(sq_dist_1_2.shape == (n_samples, 1))

if __name__ == '__main__':
        unittest.main()
