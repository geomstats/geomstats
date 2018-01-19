"""Unit tests for special euclidean group module."""

import numpy as np
import unittest

from geomstats.lie_groups import InvariantMetric
from geomstats.special_euclidean_group import SpecialEuclideanGroup


class TestSpecialEuclideanGroupMethods(unittest.TestCase):
    DIMENSION = 6
    GROUP = SpecialEuclideanGroup(dimension=DIMENSION)
    ALGEBRA_CANONICAL_INNER_PRODUCT = np.eye(6)

    LEFT_CANONICAL_METRIC = InvariantMetric(
                lie_group=GROUP,
                metric_matrix_at_identity=ALGEBRA_CANONICAL_INNER_PRODUCT,
                left_or_right='left')

    RIGHT_CANONICAL_METRIC = InvariantMetric(
                lie_group=GROUP,
                metric_matrix_at_identity=ALGEBRA_CANONICAL_INNER_PRODUCT,
                left_or_right='right')

    metric_matrix_at_identity = np.zeros([6, 6])
    metric_matrix_at_identity[0:3, 0:3] = 3 * np.eye(3)
    metric_matrix_at_identity[3:6, 3:6] = 9 * np.eye(3)

    LEFT_DIAG_METRIC = InvariantMetric(
                           lie_group=GROUP,
                           metric_matrix_at_identity=metric_matrix_at_identity,
                           left_or_right='left')
    RIGHT_DIAG_METRIC = InvariantMetric(
                           lie_group=GROUP,
                           metric_matrix_at_identity=metric_matrix_at_identity,
                           left_or_right='right')

    def test_compose(self):
        # 1. Composition by identity, on the right
        # Expect the original transformation
        transfo_1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        result_1 = self.GROUP.compose(transfo_1, self.GROUP.identity)
        expected_1 = transfo_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Composition by identity, on the left
        # Expect the original transformation
        transfo_2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        result_2 = self.GROUP.compose(self.GROUP.identity, transfo_2)
        expected_2 = transfo_2

        self.assertTrue(np.allclose(result_2, expected_2))

        # 3. Composition of translations (no rotational part)
        # Expect the sum of the translations
        transfo_a_3 = np.array([0., 0., 0., 0.4, 0.5, 0.6])
        transfo_b_3 = np.array([0., 0., 0., 0.5, 0.6, 0.7])

        result_3 = self.GROUP.compose(transfo_a_3, transfo_b_3)
        expected_3 = np.array([0., 0., 0., 0.9, 1.1, 1.3])

        self.assertTrue(np.allclose(result_3, expected_3))

    def test_compose_and_inverse(self):
        # 1. Compose transformation by its inverse on the right
        # Expect the group identity
        transfo_1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        inv_transfo_1 = self.GROUP.inverse(transfo_1)

        result_1 = self.GROUP.compose(transfo_1, inv_transfo_1)
        expected_1 = self.GROUP.identity

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose transformation by its inverse on the left
        # Expect the group identity
        transfo_2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        inv_transfo_2 = self.GROUP.inverse(transfo_2)

        result_2 = self.GROUP.compose(inv_transfo_2, transfo_2)
        expected_2 = self.GROUP.identity

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_group_log_from_identity(self):
        # 1. Group logarithm of a translation (no rotational part)
        # Expect the original translation
        rot_vec_1 = np.array([0, 0, 0])
        translation_1 = np.array([1, 0, -3])
        transfo_1 = np.concatenate([rot_vec_1, translation_1])

        result_1 = self.GROUP.group_log(ref_point=self.GROUP.identity,
                                        point=transfo_1)
        expected_1 = transfo_1

        self.assertTrue(np.allclose(expected_1, result_1))

        # 2. Group logarithm of a transformation
        # where translation is parallel to rotation axis
        # Expect the original transformation
        rot_vec_2 = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_2 = np.array([4, 0, 0])
        transfo_2 = np.concatenate([rot_vec_2, translation_2])

        result_2 = self.GROUP.group_log(ref_point=self.GROUP.identity,
                                        point=transfo_2)
        expected_2 = transfo_2

        self.assertTrue(np.allclose(expected_2, result_2))

    def test_group_exp_from_identity(self):
        # 1. Group exponential of a translation (no rotational part)
        # Expect the original translation
        rot_vec_1 = np.array([0, 0, 0])
        translation_1 = np.array([1, 0, -3])
        tangent_vec_1 = np.concatenate([rot_vec_1, translation_1])

        result_1 = self.GROUP.group_exp(ref_point=self.GROUP.identity,
                                        tangent_vec=tangent_vec_1)
        expected_1 = tangent_vec_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Group exponential of a transformation
        # where translation is parallel to rotation axis
        # Expect the original transformation
        rot_vec_2 = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_2 = np.array([4, 0, 0])
        tangent_vec_2 = np.concatenate([rot_vec_2, translation_2])

        result_2 = self.GROUP.group_exp(ref_point=self.GROUP.identity,
                                        tangent_vec=tangent_vec_2)
        expected_2 = tangent_vec_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_group_exp_and_log_from_identity(self):
        """
        Test that the group exponential from the identity
        and the group logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        # 1. Compose log then exp
        rot_vec_1 = np.array([0.01, -1., -0.8])  # NB: Regularized
        translation_1 = np.array([10., 2., 7.])
        point_1 = np.concatenate([rot_vec_1, translation_1])

        aux_1 = self.GROUP.group_log(ref_point=self.GROUP.identity,
                                     point=point_1)
        result_1 = self.GROUP.group_exp(ref_point=self.GROUP.identity,
                                        tangent_vec=aux_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([1e-10, 0., -6 * 1e-6])  # NB: Regularized
        translation_2 = np.array([-1., 27., 7.])
        point_2 = np.concatenate([rot_vec_2, translation_2])

        aux_2 = self.GROUP.group_log(ref_point=self.GROUP.identity,
                                     point=point_2)
        result_2 = self.GROUP.group_exp(ref_point=self.GROUP.identity,
                                        tangent_vec=aux_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

        # 3. Compose exp then log
        rot_vec_3 = np.array([0.01, -1., -0.8])  # NB: Regularized
        translation_3 = np.array([10., 2., 7.])
        tangent_vec_3 = np.concatenate([rot_vec_3, translation_3])

        aux_3 = self.GROUP.group_exp(ref_point=self.GROUP.identity,
                                     tangent_vec=tangent_vec_3)
        result_3 = self.GROUP.group_log(ref_point=self.GROUP.identity,
                                        point=aux_3)
        expected_3 = tangent_vec_3

        self.assertTrue(np.allclose(result_3, expected_3))

        # 4. Compose exp then log
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_4 = np.array([1e-10, 0., -6 * 1e-6])  # NB: Regularized
        translation_4 = np.array([-1., 27., 7.])
        tangent_vec_4 = np.concatenate([rot_vec_4, translation_4])

        aux_4 = self.GROUP.group_exp(ref_point=self.GROUP.identity,
                                     tangent_vec=tangent_vec_4)
        result_4 = self.GROUP.group_log(ref_point=self.GROUP.identity,
                                        point=aux_4)
        expected_4 = tangent_vec_4

        self.assertTrue(np.allclose(result_4, expected_4))

    def test_group_exp(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        rot_vec_ref_point = np.array([0., 0., 0.])
        translation_ref_point = np.array([4, -1, 10000])
        transfo_ref_point = np.concatenate([rot_vec_ref_point,
                                            translation_ref_point])
        # 1. Tangent vector is a translation (no infinitesimal rotational part)
        # Expect the sum of the translation
        # with the translation of the reference point
        rot_vec_1 = np.array([0., 0., 0.])
        translation_1 = np.array([1, 0, -3])
        tangent_vec_1 = np.concatenate([rot_vec_1, translation_1])

        result_1 = self.GROUP.group_exp(ref_point=transfo_ref_point,
                                        tangent_vec=tangent_vec_1)
        expected_1 = np.concatenate([np.array([0., 0., 0.]),
                                     np.array([5, -1, 9997])])
        self.assertTrue(np.allclose(result_1, expected_1))

    def test_group_log(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        rot_vec_ref_point = np.array([0., 0., 0.])
        translation_ref_point = np.array([4., 0., 0.])
        transfo_ref_point = np.concatenate([rot_vec_ref_point,
                                            translation_ref_point])

        # 1. Point is a translation (no rotational part)
        # Expect the difference of the translation
        # by the translation of the reference point
        rot_vec_1 = np.array([0., 0., 0.])
        translation_1 = np.array([5., 8., -3.2])
        point_1 = np.concatenate([rot_vec_1,
                                  translation_1])

        expected_1 = np.concatenate([np.array([0., 0., 0.]),
                                     np.array([1., 8., -3.2])])

        result_1 = self.GROUP.group_log(ref_point=transfo_ref_point,
                                        point=point_1)

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_group_exp_and_log(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        rot_vec_ref_point = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_ref_point = np.array([4, -1, 2])
        transfo_ref_point = np.concatenate([rot_vec_ref_point,
                                            translation_ref_point])

        # 1. Compose log then exp
        rot_vec_1 = np.array([-1.2, 0.9, 0.9])  # NB: Regularized
        translation_1 = np.array([5, 5, 5])
        point_1 = np.concatenate([rot_vec_1,
                                  translation_1])

        aux_1 = self.GROUP.group_log(ref_point=transfo_ref_point,
                                     point=point_1)
        result_1 = self.GROUP.group_exp(ref_point=transfo_ref_point,
                                        tangent_vec=aux_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([-1e-7, 0., -7 * 1e-8])  # NB: Regularized
        translation_2 = np.array([6, 5, 9])
        point_2 = np.concatenate([rot_vec_2,
                                  translation_2])

        aux_2 = self.GROUP.group_log(ref_point=transfo_ref_point,
                                     point=point_2)
        result_2 = self.GROUP.group_exp(ref_point=transfo_ref_point,
                                        tangent_vec=aux_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_riemannian_left_exp_from_id(self):
        # Riemannian left-invariant metric given by
        # the canonical inner product on the lie algebra
        # Expect the identity function
        # because we use the riemannian left logarithm with canonical
        # inner product to parameterize the transformations

        # 1. General case
        tangent_rot_vec_1 = np.array([1., 1., 1.])  # NB: Regularized
        tangent_translation_1 = np.array([1., 0., -3.])
        tangent_vec_1 = np.concatenate([tangent_rot_vec_1,
                                        tangent_translation_1])

        result_1 = self.LEFT_CANONICAL_METRIC.riemannian_exp_from_identity(
                                                             tangent_vec_1)
        expected_1 = tangent_vec_1

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_riemannian_left_log_from_id(self):
        # Riemannian left-invariant metric given by
        # the canonical inner product on the lie algebra
        # Expect the identity function
        # because we use the riemannian left logarithm with canonical
        # inner product to parameterize the transformations

        # 1. General case
        rot_vec_1 = np.array([0.1, 1, 0.9])  # NB: Regularized
        translation_1 = np.array([1, -19, -3])
        transfo_1 = np.concatenate([rot_vec_1, translation_1])

        expected_1 = transfo_1
        result_1 = self.LEFT_CANONICAL_METRIC.riemannian_log_from_identity(
                                                                 transfo_1)

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([1e-8, 0., 1e-9])  # NB: Regularized
        translation_2 = np.array([10000, -5.9, -93])
        transfo_2 = np.concatenate([rot_vec_2, translation_2])

        expected_2 = transfo_2
        result_2 = self.LEFT_CANONICAL_METRIC.riemannian_log_from_identity(
                                                                 transfo_2)

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_riemannian_left_exp_and_log_from_id(self):
        """
        Test that the riemannian left exponential from the identity
        and the riemannian left logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        # 1. Compose log then exp
        # Canonical inner product on the lie algebra
        rot_vec_1 = np.array([-1., 0.5, -0.12])  # NB: Regularized
        translation_1 = np.array([-91., -7., 0.007])
        point_1 = np.concatenate([rot_vec_1, translation_1])

        riem_log_1 = self.LEFT_CANONICAL_METRIC.riemannian_log_from_identity(
                                                  point=point_1)
        result_1 = self.LEFT_CANONICAL_METRIC.riemannian_exp_from_identity(
                                         tangent_vec=riem_log_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Canonical inner product on the lie algebra
        rot_vec_2 = np.array([1e-15, 0., 5 * 1e-6])  # NB: Regularized
        translation_2 = np.array([-1., 27., 7.])
        point_2 = np.concatenate([rot_vec_2, translation_2])

        riem_log_2 = self.LEFT_CANONICAL_METRIC.riemannian_log_from_identity(
                                                                     point_2)
        result_2 = self.LEFT_CANONICAL_METRIC.riemannian_exp_from_identity(
                                                                riem_log_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

        # 3. Compose log then exp
        # Block diagonal inner product
        rot_vec_3 = np.array([-1., 0.5, -0.12])  # NB: Regularized
        translation_3 = np.array([-91., -7., 0.007])
        point_3 = np.concatenate([rot_vec_3, translation_3])

        aux_3 = self.LEFT_DIAG_METRIC.riemannian_log_from_identity(point_3)
        result_3 = self.LEFT_DIAG_METRIC.riemannian_exp_from_identity(aux_3)
        expected_3 = point_3

        self.assertTrue(np.allclose(result_3, expected_3))

        # 4. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Block diagonal inner product
        rot_vec_4 = np.array([1e-15, 0., 5 * 1e-6])  # NB: Regularized
        translation_4 = np.array([-1., 27., 7.])
        point_4 = np.concatenate([rot_vec_4, translation_4])

        aux_4 = self.LEFT_DIAG_METRIC.riemannian_log_from_identity(point_4)
        result_4 = self.LEFT_DIAG_METRIC.riemannian_exp_from_identity(aux_4)
        expected_4 = point_4

        self.assertTrue(np.allclose(result_4, expected_4))

    def test_riemannian_right_exp_and_log_from_id(self):
        """
        Test that the riemannian right exponential from the identity
        and the riemannian right logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        # 1. Compose log then exp
        # Canonical inner product on the lie algebra
        rot_vec_1 = np.array([-1., 0.5, -0.12])  # NB: Regularized
        translation_1 = np.array([-91., -7., 0.007])
        point_1 = np.concatenate([rot_vec_1, translation_1])

        aux_1 = self.RIGHT_CANONICAL_METRIC.riemannian_log_from_identity(
                                                                  point_1)
        result_1 = self.RIGHT_CANONICAL_METRIC.riemannian_exp_from_identity(
                                                                    aux_1)

        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Canonical inner product on the lie algebra
        rot_vec_2 = np.array([1e-15, 0., 5 * 1e-6])  # NB: Regularized
        translation_2 = np.array([-1., 27., 7.])
        point_2 = np.concatenate([rot_vec_2, translation_2])

        aux_2 = self.RIGHT_CANONICAL_METRIC.riemannian_log_from_identity(
                                                                   point_2)
        result_2 = self.RIGHT_CANONICAL_METRIC.riemannian_exp_from_identity(
                                                                     aux_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

        # 3. Compose log then exp
        # Block diagonal inner product
        rot_vec_3 = np.array([-1., 0.5, -0.12])  # NB: Regularized
        translation_3 = np.array([-91., -7., 0.007])
        point_3 = np.concatenate([rot_vec_3, translation_3])

        aux_3 = self.RIGHT_DIAG_METRIC.riemannian_log_from_identity(point_3)
        result_3 = self.RIGHT_DIAG_METRIC.riemannian_exp_from_identity(aux_3)
        expected_3 = point_3

        self.assertTrue(np.allclose(result_3, expected_3))

        # 4. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Block diagonal inner product
        rot_vec_4 = np.array([1e-15, 0., 5 * 1e-6])  # NB: Regularized
        translation_4 = np.array([-1., 27., 7.])
        point_4 = np.concatenate([rot_vec_4, translation_4])

        aux_4 = self.RIGHT_DIAG_METRIC.riemannian_log_from_identity(point_4)
        result_4 = self.RIGHT_DIAG_METRIC.riemannian_exp_from_identity(aux_4)
        expected_4 = point_4

        self.assertTrue(np.allclose(result_4, expected_4))

    def test_riemannian_left_exp(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        rot_vec_ref_point = np.array([0., 0., 0.])
        translation_ref_point = np.array([4, -1, 10000])
        transfo_ref_point = np.concatenate([rot_vec_ref_point,
                                            translation_ref_point])

        # 1. Tangent vector is a translation (no infinitesimal rotational part)
        # Expect the sum of the translation
        # with the translation of the reference point
        rot_vec_1 = np.array([0., 0., 0.])
        translation_1 = np.array([1, 0, -3])
        tangent_vec_1 = np.concatenate([rot_vec_1, translation_1])

        result_1 = self.LEFT_CANONICAL_METRIC.riemannian_exp(
                                         ref_point=transfo_ref_point,
                                         tangent_vec=tangent_vec_1)
        expected_1 = np.concatenate([np.array([0., 0., 0.]),
                                     np.array([5, -1, 9997])])
        self.assertTrue(np.allclose(result_1, expected_1))

    def test_riemannian_left_log(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        rot_vec_ref_point = np.array([0., 0., 0.])
        translation_ref_point = np.array([4., 0., 0.])
        transfo_ref_point = np.concatenate([rot_vec_ref_point,
                                            translation_ref_point])

        # 1. Point is a translation (no rotational part)
        # Expect the difference of the translation
        # by the translation of the reference point
        rot_vec_1 = np.array([0., 0., 0.])
        translation_1 = np.array([-1., -1., -1.2])
        point_1 = np.concatenate([rot_vec_1,
                                  translation_1])

        expected_1 = np.concatenate([np.array([0., 0., 0.]),
                                     np.array([-5., -1., -1.2])])

        result_1 = self.LEFT_CANONICAL_METRIC.riemannian_log(
                                       ref_point=transfo_ref_point,
                                       point=point_1)

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_riemannian_left_exp_and_log(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        rot_vec_ref_point = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_ref_point = np.array([4, -1, 2])
        transfo_ref_point = np.concatenate([rot_vec_ref_point,
                                            translation_ref_point])

        # 1. Compose log then exp
        # Canonical inner product on the lie algebra
        rot_vec_1 = np.array([-1.2, 0.9, 0.9])  # NB: Regularized
        translation_1 = np.array([5, 5, 5])
        point_1 = np.concatenate([rot_vec_1,
                                  translation_1])

        aux_1 = self.LEFT_CANONICAL_METRIC.riemannian_log(
                                          ref_point=transfo_ref_point,
                                          point=point_1)
        result_1 = self.LEFT_CANONICAL_METRIC.riemannian_exp(
                                          ref_point=transfo_ref_point,
                                          tangent_vec=aux_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Canonical inner product on the lie algebra
        rot_vec_2 = np.array([-1e-7, 0., -7*1e-8])  # NB: Regularized
        translation_2 = np.array([6, 5, 9])
        point_2 = np.concatenate([rot_vec_2,
                                  translation_2])

        aux_2 = self.LEFT_CANONICAL_METRIC.riemannian_log(
                                        ref_point=transfo_ref_point,
                                        point=point_2)
        result_2 = self.LEFT_CANONICAL_METRIC.riemannian_exp(
                                        ref_point=transfo_ref_point,
                                        tangent_vec=aux_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

        # 3. Compose log then exp
        # Block diagonal inner product
        rot_vec_3 = np.array([-1.2, 0.9, 0.9])  # NB: Regularized
        translation_3 = np.array([5, 5, 5])
        point_3 = np.concatenate([rot_vec_3,
                                  translation_3])

        aux_3 = self.LEFT_DIAG_METRIC.riemannian_log(
                                      ref_point=transfo_ref_point,
                                      point=point_3)
        result_3 = self.LEFT_DIAG_METRIC.riemannian_exp(
                                      ref_point=transfo_ref_point,
                                      tangent_vec=aux_3)
        expected_3 = point_3

        self.assertTrue(np.allclose(result_3, expected_3))

        # 4. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Block diagonal inner product
        rot_vec_4 = np.array([-1e-7, 0., -7*1e-8])  # NB: Regularized
        translation_4 = np.array([6, 5, 9])
        point_4 = np.concatenate([rot_vec_4,
                                  translation_4])

        aux_4 = self.LEFT_DIAG_METRIC.riemannian_log(
                                      ref_point=transfo_ref_point,
                                      point=point_4)
        result_4 = self.LEFT_DIAG_METRIC.riemannian_exp(
                                         ref_point=transfo_ref_point,
                                         tangent_vec=aux_4)
        expected_4 = point_4

        self.assertTrue(np.allclose(result_4, expected_4))

    def test_riemannian_right_exp_and_log(self):
        """
        Test that the riemannian right exponential and the
        riemannian right logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        rot_vec_ref_point = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_ref_point = np.array([4, -1, 2])
        transfo_ref_point = np.concatenate([rot_vec_ref_point,
                                            translation_ref_point])

        # 1. Compose log then exp
        rot_vec_1 = np.array([-1.2, 0.9, 0.9])  # NB: Regularized
        translation_1 = np.array([5, 5, 5])
        point_1 = np.concatenate([rot_vec_1,
                                  translation_1])

        aux_1 = self.RIGHT_CANONICAL_METRIC.riemannian_log(
                                      ref_point=transfo_ref_point,
                                      point=point_1)
        result_1 = self.RIGHT_CANONICAL_METRIC.riemannian_exp(
                                      ref_point=transfo_ref_point,
                                      tangent_vec=aux_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([-1e-7, 0., -7*1e-8])  # NB: Regularized
        translation_2 = np.array([6, 5, 9])
        point_2 = np.concatenate([rot_vec_2,
                                  translation_2])

        aux_2 = self.RIGHT_CANONICAL_METRIC.riemannian_log(
                                      ref_point=transfo_ref_point,
                                      point=point_2)
        result_2 = self.RIGHT_CANONICAL_METRIC.riemannian_exp(
                                      ref_point=transfo_ref_point,
                                      tangent_vec=aux_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

        # 3. Compose log then exp
        # Block diagonal inner product
        rot_vec_3 = np.array([-1.2, 0.9, 0.9])  # NB: Regularized
        translation_3 = np.array([5, 5, 5])
        point_3 = np.concatenate([rot_vec_3,
                                  translation_3])

        inner_product = np.zeros([6, 6])
        inner_product[0:3, 0:3] = 3 * np.eye(3)
        inner_product[3:6, 3:6] = 9 * np.eye(3)

        aux_3 = self.RIGHT_CANONICAL_METRIC.riemannian_log(
                                      ref_point=transfo_ref_point,
                                      point=point_3)
        result_3 = self.RIGHT_CANONICAL_METRIC.riemannian_exp(
                                      ref_point=transfo_ref_point,
                                      tangent_vec=aux_3)
        expected_3 = point_3

        self.assertTrue(np.allclose(result_3, expected_3))

        # 4. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Block diagonal inner product
        rot_vec_4 = np.array([-1e-7, 0., -7*1e-8])  # NB: Regularized
        translation_4 = np.array([6, 5, 9])
        point_4 = np.concatenate([rot_vec_4,
                                  translation_4])

        inner_product = np.zeros([6, 6])
        inner_product[0:3, 0:3] = 3 * np.eye(3)
        inner_product[3:6, 3:6] = 9 * np.eye(3)

        aux_4 = self.RIGHT_CANONICAL_METRIC.riemannian_log(
                                      ref_point=transfo_ref_point,
                                      point=point_4)
        result_4 = self.RIGHT_CANONICAL_METRIC.riemannian_exp(
                                      ref_point=transfo_ref_point,
                                      tangent_vec=aux_4)
        expected_4 = point_4

        self.assertTrue(np.allclose(result_4, expected_4))


if __name__ == '__main__':
        unittest.main()
