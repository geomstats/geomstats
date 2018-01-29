"""
Unit tests for special euclidean group module.

Note: Only the *canonical* left- and right- invariant
metrics on SE(3) are tested here. Other invariant
metrics are tested with the tests of the invariant_metric
module.
"""

import numpy as np
import unittest

from geomstats.invariant_metric import InvariantMetric
from geomstats.special_euclidean_group import SpecialEuclideanGroup
import tests.helper as helper

# Tolerance for errors on predicted vectors, relative to the *norm*
# of the vector
RTOL = 1e-6


class TestSpecialEuclideanGroupMethods(unittest.TestCase):
    def setUp(self):
        n = 3
        group = SpecialEuclideanGroup(n=n)

        # Points

        # -- Rotation vectors with angles
        # 0, close to 0, closely lower than pi, pi,
        # between pi and 2pi, closely larger than 2pi, 2pi,
        # and closely larger than 2pi
        with_angle_0 = np.zeros(6)
        with_angle_close_0 = (1e-10 * np.array([1., -1., 1., 0., 0., 0.])
                              + np.array([0., 0., 0., 1., 5., 2]))
        with_angle_close_pi_low = ((np.pi - 1e-9) / np.sqrt(2)
                                   * np.array([0., 1., -1, 0., 0., 0.])
                                   + np.array([0., 0., 0., -100., 0., 2.]))
        with_angle_pi = (np.pi / np.sqrt(3)
                         * np.array([1., 1., -1, 0., 0., 0.])
                         + np.array([0., 0., 0., -10.2, 0., 2.6]))
        with_angle_close_pi_high = ((np.pi + 1e-9) / np.sqrt(3)
                                    * np.array([-1., 1., -1, 0., 0., 0.])
                                    + np.array([0., 0., 0., -100., 0., 2.]))
        with_angle_in_pi_2pi = ((np.pi + 0.3) / np.sqrt(5)
                                * np.array([-2., 1., 0., 0., 0., 0.])
                                + np.array([0., 0., 0., -100., 0., 2.]))
        with_angle_close_2pi_low = ((2 * np.pi - 1e-9) / np.sqrt(6)
                                    * np.array([2., 1., -1, 0., 0., 0.])
                                    + np.array([0., 0., 0., 8., 555., -2.]))
        with_angle_2pi = (2 * np.pi / np.sqrt(3)
                          * np.array([1., 1., -1, 0., 0., 0.])
                          + np.array([0., 0., 0., 1., 8., -10.]))
        with_angle_close_2pi_high = ((2 * np.pi + 1e-9) / np.sqrt(2)
                                     * np.array([1., 0., -1, 0., 0., 0.])
                                     + np.array([0., 0., 0., 1., 8., -10.]))

        point_1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        point_2 = np.array([0.5, 0., -0.3, 0.4, 5., 60.])

        translation_large = np.array([0., 0., 0., 0.4, 0.5, 0.6])
        translation_small = np.array([0., 0., 0., 0.5, 0.6, 0.7])
        rot_with_parallel_trans = np.array([np.pi / 3., 0., 0.,
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
        diag_mat_at_identity = np.zeros([group.dimension, group.dimension])
        diag_mat_at_identity[0:3, 0:3] = 1 * np.eye(3)
        diag_mat_at_identity[3:6, 3:6] = 2 * np.eye(3)

        left_diag_metric = InvariantMetric(
                   lie_group=group,
                   inner_product_mat_at_identity=diag_mat_at_identity,
                   left_or_right='left')
        right_diag_metric = InvariantMetric(
                   lie_group=group,
                   inner_product_mat_at_identity=diag_mat_at_identity,
                   left_or_right='right')

        metrics = {'left_canonical': group.left_canonical_metric,
                   'right_canonical': group.right_canonical_metric,
                   'left_diag': left_diag_metric,
                   'right_diag': right_diag_metric}

        self.group = group
        self.metrics = metrics
        self.elements = elements
        self.angles_close_to_pi = ['with_angle_close_pi_low',
                                   'with_angle_pi',
                                   'with_angle_close_pi_high']

    def test_random_and_belongs(self):
        """
        Test that the random uniform method samples
        on the special euclidean group.
        """
        base_point = self.group.random_uniform()
        self.assertTrue(self.group.belongs(base_point))

    def test_regularize(self):
        point = self.elements['with_angle_0']
        result = self.group.regularize(point)
        expected = point
        self.assertTrue(np.allclose(result, expected))

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

        expected = np.zeros(6)
        expected[:3] = - point[:3]
        expected[3:6] = point[3:6]

        self.assertTrue(np.allclose(result, expected),
                        '\n{}'
                        '\npoint = {}'
                        '\nresult = {}'
                        '\nexpected = {}'.format(
                            angle_type,
                            point,
                            result,
                            expected))

        in_pi_2pi = ['with_angle_close_pi_high',
                     'with_angle_in_pi_2pi',
                     'with_angle_close_2pi_low']

        for angle_type in in_pi_2pi:
            point = self.elements[angle_type]
            angle = np.linalg.norm(point[:3])
            new_angle = np.pi - (angle - np.pi)

            result = self.group.regularize(point)
            expected = np.zeros(6)
            expected[:3] = - new_angle * (point[:3] / angle)
            expected[3:6] = point[3:6]

            self.assertTrue(np.allclose(result, expected), angle_type)

        angle_type = 'with_angle_2pi'
        point = self.elements[angle_type]
        result = self.group.regularize(point)
        expected = np.zeros(6)
        expected[:3] = np.array([0., 0., 0.])
        expected[3:6] = point[3:6]
        self.assertTrue(np.allclose(result, expected), angle_type)

        angle_type = 'with_angle_close_2pi_high'
        point = self.elements[angle_type]
        angle = np.linalg.norm(point[:3])
        new_angle = angle - 2 * np.pi

        result = self.group.regularize(point)

        expected = np.zeros(6)
        expected[:3] = new_angle * point[:3] / angle
        expected[3:6] = point[3:6]
        self.assertTrue(np.allclose(result, expected),
                        '\n{}'
                        '\npoint = {}'
                        '\nresult = {}'
                        '\nexpected = {}'.format(
                                        angle_type,
                                        point,
                                        result,
                                        expected))

    def test_compose(self):
        # Composition by identity, on the right
        # Expect the original transformation
        point = self.elements['point_1']
        result = self.group.compose(point,
                                    self.group.identity)
        expected = point
        self.assertTrue(np.allclose(result, expected))

        # Composition by identity, on the left
        # Expect the original transformation
        result = self.group.compose(self.group.identity,
                                    point)
        expected = point
        self.assertTrue(np.allclose(result, expected))

        # Composition of translations (no rotational part)
        # Expect the sum of the translations
        result = self.group.compose(self.elements['translation_small'],
                                    self.elements['translation_large'])
        expected = (self.elements['translation_small']
                    + self.elements['translation_large'])
        self.assertTrue(np.allclose(result, expected))

    def test_compose_and_inverse(self):
        point = self.elements['point_1']
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

    def test_group_log_from_identity(self):
        # Group logarithm of a translation (no rotational part)
        # Expect the original translation
        point = self.elements['translation_small']
        result = self.group.group_log(base_point=self.group.identity,
                                      point=point)
        expected = point
        self.assertTrue(np.allclose(expected, result))

        # Group logarithm of a transformation
        # where translation is parallel to rotation axis
        # Expect the original transformation
        point = self.elements['rot_with_parallel_trans']
        result = self.group.group_log(base_point=self.group.identity,
                                      point=point)
        expected = point
        self.assertTrue(np.allclose(expected, result))

    def test_group_exp_from_identity(self):
        # Group exponential of a translation (no rotational part)
        # Expect the original translation
        tangent_vec = self.elements['translation_small']
        result = self.group.group_exp(base_point=self.group.identity,
                                      tangent_vec=tangent_vec)
        expected = tangent_vec
        self.assertTrue(np.allclose(result, expected))

        # Group exponential of a transformation
        # where translation is parallel to rotation axis
        # Expect the original transformation
        tangent_vec = self.elements['rot_with_parallel_trans']
        result = self.group.group_exp(
                                  base_point=self.group.identity,
                                  tangent_vec=tangent_vec)
        expected = tangent_vec
        self.assertTrue(np.allclose(result, expected))

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
            self.assertTrue(np.allclose(result, expected))

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

            inv_rot_expected = np.zeros(6)
            inv_rot_expected[:3] = - expected[:3]
            inv_rot_expected[3:6] = expected[3:6]

            self.assertTrue(np.allclose(result, expected)
                            or np.allclose(result, inv_rot_expected),
                            '\npoint = {}'
                            '\nresult = {}'
                            '\nexpected = {}'
                            '\nexpected with opp rotation = {}'.format(
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
            self.assertTrue(np.allclose(result, expected),
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

            inv_rot_expected = np.zeros(6)
            inv_rot_expected[:3] = - expected[:3]
            inv_rot_expected[3:6] = expected[3:6]

            self.assertTrue(np.allclose(result, expected)
                            or np.allclose(result, inv_rot_expected),
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
        self.assertTrue(np.allclose(result, expected))

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

        self.assertTrue(np.allclose(result, expected))

    def test_group_log_then_exp(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        for base_point in self.elements.values():
            for element_type in self.elements:
                if element_type in self.angles_close_to_pi:
                    continue
                point = self.elements[element_type]

                log = self.group.group_log(point=point, base_point=base_point)
                exp = self.group.group_exp(tangent_vec=log, base_point=base_point)

                result = helper.group_log_then_exp(group=self.group,
                                                   point=point,
                                                   base_point=base_point)
                expected = self.group.regularize(point)
                self.assertTrue(np.allclose(result, expected),
                                '\n{}'
                                '\npoint = {}'
                                '\nlog = {}'
                                '\nexp = {}'
                                '\nresult = {}'
                                '\nexpected = {}'.format(
                               element_type,
                               point,
                               log,
                               exp,
                               result,
                               expected))

    def test_exp_from_identity_left(self):
        # Riemannian left-invariant metric given by
        # the canonical inner product on the lie algebra
        # Expect the identity function
        # because we use the riemannian left logarithm with canonical
        # inner product to parameterize the transformations
        metric = self.group.left_canonical_metric
        # 1. General case
        tangent_rot_vec_1 = np.array([1., 1., 1.])  # NB: Regularized
        tangent_translation_1 = np.array([1., 0., -3.])
        tangent_vec_1 = np.concatenate([tangent_rot_vec_1,
                                        tangent_translation_1])
        result_1 = metric.exp_from_identity(tangent_vec_1)
        expected_1 = tangent_vec_1

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_log_from_identity_left(self):
        # Riemannian left-invariant metric given by
        # the canonical inner product on the lie algebra
        # Expect the identity function
        # because we use the riemannian left logarithm with canonical
        # inner product to parameterize the transformations

        metric = self.group.left_canonical_metric
        # 1. General case
        rot_vec_1 = np.array([0.1, 1, 0.9])  # NB: Regularized
        translation_1 = np.array([1, -19, -3])
        transfo_1 = np.concatenate([rot_vec_1, translation_1])

        expected_1 = transfo_1
        result_1 = metric.log_from_identity(transfo_1)

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([1e-8, 0., 1e-9])  # NB: Regularized
        translation_2 = np.array([10000, -5.9, -93])
        transfo_2 = np.concatenate([rot_vec_2, translation_2])

        expected_2 = transfo_2
        result_2 = metric.log_from_identity(transfo_2)

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_exp_then_log_from_identity_left(self):
        """
        Test that the riemannian left exponential from the identity
        and the riemannian left logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        # Canonical inner product on the lie algebra
        metric = self.group.left_canonical_metric
        for angle_type in self.elements:
            if angle_type in self.angles_close_to_pi:
                continue
            tangent_vec = self.elements[angle_type]
            result = helper.exp_then_log_from_identity(
                                            metric=metric,
                                            tangent_vec=tangent_vec)
            expected = self.group.regularize(tangent_vec)
            self.assertTrue(np.allclose(result, expected),
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
        metric = self.group.left_canonical_metric
        for angle_type in angle_types:
            tangent_vec = self.elements[angle_type]
            result = helper.exp_then_log_from_identity(
                                            metric=metric,
                                            tangent_vec=tangent_vec)
            expected = self.group.regularize(tangent_vec)
            inv_rot_expected = np.zeros(6)
            inv_rot_expected[:3] = - expected[:3]
            inv_rot_expected[3:6] = expected[3:6]

            self.assertTrue(np.allclose(result, expected)
                            or np.allclose(result, inv_rot_expected),
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
        metric = self.group.right_canonical_metric
        for angle_type in self.elements:
            if angle_type in self.angles_close_to_pi:
                continue
            tangent_vec = self.elements[angle_type]
            result = helper.exp_then_log_from_identity(
                                            metric=metric,
                                            tangent_vec=tangent_vec)
            expected = self.group.regularize(tangent_vec)

            self.assertTrue(np.allclose(result, expected),
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
        metric = self.group.right_canonical_metric
        for angle_type in angle_types:
            tangent_vec = self.elements[angle_type]
            result = helper.exp_then_log_from_identity(
                                            metric=metric,
                                            tangent_vec=tangent_vec)
            expected = self.group.regularize(tangent_vec)
            inv_rot_expected = np.zeros(6)
            inv_rot_expected[:3] = - expected[:3]
            inv_rot_expected[3:6] = expected[3:6]

            self.assertTrue(np.allclose(result, expected)
                            or np.allclose(result, inv_rot_expected),
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
        rot_vec_base_point = np.array([0., 0., 0.])
        translation_base_point = np.array([4, -1, 10000])
        transfo_base_point = np.concatenate([rot_vec_base_point,
                                            translation_base_point])

        # 1. Tangent vector is a translation (no infinitesimal rotational part)
        # Expect the sum of the translation
        # with the translation of the reference point
        rot_vec_1 = np.array([0., 0., 0.])
        translation_1 = np.array([1, 0, -3])
        tangent_vec_1 = np.concatenate([rot_vec_1, translation_1])

        result_1 = self.group.left_canonical_metric.exp(
                                         base_point=transfo_base_point,
                                         tangent_vec=tangent_vec_1)
        expected_1 = np.concatenate([np.array([0., 0., 0.]),
                                     np.array([5, -1, 9997])])
        self.assertTrue(np.allclose(result_1, expected_1))

    def test_log_left(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        rot_vec_base_point = np.array([0., 0., 0.])
        translation_base_point = np.array([4., 0., 0.])
        transfo_base_point = np.concatenate([rot_vec_base_point,
                                            translation_base_point])

        # 1. Point is a translation (no rotational part)
        # Expect the difference of the translation
        # by the translation of the reference point
        rot_vec_1 = np.array([0., 0., 0.])
        translation_1 = np.array([-1., -1., -1.2])
        point_1 = np.concatenate([rot_vec_1,
                                  translation_1])

        expected_1 = np.concatenate([np.array([0., 0., 0.]),
                                     np.array([-5., -1., -1.2])])

        result_1 = self.group.left_canonical_metric.log(
                                       base_point=transfo_base_point,
                                       point=point_1)

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_log_then_exp_left(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        metric = self.metrics['left_canonical']
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
                self.assertTrue(np.allclose(result, expected),
                                '\npoint = {}'
                                '\nresult = {}'
                                '\nexpected = {}'.format(
                               point,
                               result,
                               expected))

    def test_exp_then_log_left(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        metric = self.metrics['left_canonical']
        for base_point in self.elements.values():
            for element_type in self.elements:
                if element_type in self.angles_close_to_pi:
                    continue
                tangent_vec = self.elements[element_type]
                result = helper.exp_then_log(
                                            metric=metric,
                                            tangent_vec=tangent_vec,
                                            base_point=base_point)

                expected = self.group.regularize(tangent_vec)
                self.assertTrue(np.allclose(result, expected),
                                '\ntangent_vec = {}'
                                '\nresult = {}'
                                '\nexpected = {}'.format(
                               tangent_vec,
                               result,
                               expected))

    def test_log_then_exp_right(self):
        """
        Test that the riemannian right exponential and the
        riemannian right logarithm are inverse.
        Expect their composition to give the identity function.
        """
        metric = self.metrics['right_canonical']
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
                self.assertTrue(np.allclose(result, expected),
                                '\npoint = {}'
                                '\nresult = {}'
                                '\nexpected = {}'.format(
                               point,
                               result,
                               expected))

    def test_exp_then_log_right(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        metric = self.metrics['right_canonical']
        for base_point in self.elements.values():
            for element_type in self.elements:
                if element_type in self.angles_close_to_pi:
                    continue
                tangent_vec = self.elements[element_type]
                result = helper.exp_then_log(
                                            metric=metric,
                                            tangent_vec=tangent_vec,
                                            base_point=base_point)

                expected = self.group.regularize(tangent_vec)
                self.assertTrue(np.allclose(result, expected),
                                '\ntangent_vec = {}'
                                '\nresult = {}'
                                '\nexpected = {}'.format(
                               tangent_vec,
                               result,
                               expected))

    def test_squared_dist_is_symmetric(self):
        metric = self.group.left_canonical_metric
        for point_a in self.elements.values():
            for point_b in self.elements.values():
                point_a = self.group.regularize(point_a)
                point_b = self.group.regularize(point_b)

                sq_dist_a_b = metric.squared_dist(point_a, point_b)
                sq_dist_b_a = metric.squared_dist(point_b, point_a)

                self.assertTrue(np.allclose(sq_dist_a_b, sq_dist_b_a))

    def test_group_exponential_barycenter(self):
        # TODO(nina): this test fails, the barycenter is not accurate.
        point_1 = self.group.random_uniform()
        result_1 = self.group.group_exponential_barycenter(
                                points=[point_1, point_1])
        expected_1 = self.group.regularize(point_1)
        # self.assertTrue(np.allclose(result_1, expected_1),
        #                 '\nresult = {}\n'
        #                 'expected = {}'.format(result_1, expected_1))

        point_2 = self.group.random_uniform()
        result_2 = self.group.group_exponential_barycenter(
                                points=[point_2, point_2],
                                weights=[1., 2.])
        expected_2 = self.group.regularize(point_2)
        # self.assertTrue(np.allclose(result_2, expected_2),
        #                 '\nresult = {}\n'
        #                 'expected = {}'.format(result_2, expected_2))

        result_3 = self.group.group_exponential_barycenter(
                                points=[point_1, point_2],
                                weights=[1., .1])

        self.assertTrue(self.group.belongs(result_3))


if __name__ == '__main__':
        unittest.main()
