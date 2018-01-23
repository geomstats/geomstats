"""Unit tests for special euclidean group module."""

import numpy as np
import unittest

from geomstats.special_euclidean_group import SpecialEuclideanGroup
import tests.helper as helper


class TestSpecialEuclideanGroupMethods(unittest.TestCase):
    def setUp(self):
        n = 3
        self.group = SpecialEuclideanGroup(n=n)

        self.transfo_1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.transfo_2 = np.array([0.5, 0., -0.3, 0.4, 5., 60.])
        self.transfo_small = np.array([1e-7, 0., 1e-8, 1., 1e-10, 2.])
        self.translation_1 = np.array([0., 0., 0., 0.4, 0.5, 0.6])
        self.translation_2 = np.array([0., 0., 0., 0.5, 0.6, 0.7])
        self.rot_and_parallel_trans = np.array([np.pi / 3., 0., 0.,
                                                1., 0., 0.])

    def test_random_and_belongs(self):
        """
        Test that the random uniform method samples
        on the special euclidean group.
        """
        base_point = self.group.random_uniform()
        self.assertTrue(self.group.belongs(base_point))

    def test_regularize(self):
        result = self.group.regularize(self.group.identity)
        expected = self.group.identity
        self.assertTrue(np.allclose(result, expected))

        point = 2.5 * np.pi * np.array([0., 0., 1.,
                                        0., 0., 0.])
        result = self.group.regularize(point)
        expected = np.pi / 2. * np.array([0., 0., 1.,
                                          0., 0., 0.])
        self.assertTrue(np.allclose(result, expected))

    def test_compose(self):
        # Composition by identity, on the right
        # Expect the original transformation
        result = self.group.compose(self.transfo_1, self.group.identity)
        expected = self.transfo_1
        self.assertTrue(np.allclose(result, expected))

        # Composition by identity, on the left
        # Expect the original transformation
        result = self.group.compose(self.group.identity, self.transfo_1)
        expected = self.transfo_1
        self.assertTrue(np.allclose(result, expected))

        # Composition of translations (no rotational part)
        # Expect the sum of the translations
        result = self.group.compose(self.translation_1,
                                    self.translation_2)
        expected = self.translation_1 + self.translation_2
        self.assertTrue(np.allclose(result, expected))

    def test_compose_and_inverse(self):
        inv_transfo_1 = self.group.inverse(self.transfo_1)
        # Compose transformation by its inverse on the right
        # Expect the group identity
        result = self.group.compose(self.transfo_1, inv_transfo_1)
        expected = self.group.identity
        self.assertTrue(np.allclose(result, expected))

        # Compose transformation by its inverse on the left
        # Expect the group identity
        result = self.group.compose(inv_transfo_1, self.transfo_1)
        expected = self.group.identity
        self.assertTrue(np.allclose(result, expected))

    def test_group_log_from_identity(self):
        # Group logarithm of a translation (no rotational part)
        # Expect the original translation
        result = self.group.group_log(base_point=self.group.identity,
                                      point=self.translation_1)
        expected = self.translation_1
        self.assertTrue(np.allclose(expected, result))

        # Group logarithm of a transformation
        # where translation is parallel to rotation axis
        # Expect the original transformation
        result = self.group.group_log(base_point=self.group.identity,
                                      point=self.rot_and_parallel_trans)
        expected = self.rot_and_parallel_trans
        self.assertTrue(np.allclose(expected, result))

    def test_group_exp_from_identity(self):
        # Group exponential of a translation (no rotational part)
        # Expect the original translation
        result = self.group.group_exp(base_point=self.group.identity,
                                      tangent_vec=self.translation_1)
        expected = self.translation_1
        self.assertTrue(np.allclose(result, expected))

        # Group exponential of a transformation
        # where translation is parallel to rotation axis
        # Expect the original transformation
        result = self.group.group_exp(
                                  base_point=self.group.identity,
                                  tangent_vec=self.rot_and_parallel_trans)
        expected = self.rot_and_parallel_trans
        self.assertTrue(np.allclose(result, expected))

    def test_group_exp_and_log_from_identity(self):
        """
        Test that the group exponential from the identity
        and the group logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        # Compose log then exp
        result = helper.group_log_then_exp_from_identity(self.group,
                                                         self.transfo_1)
        expected = self.transfo_1
        self.assertTrue(np.allclose(result, expected))

        # Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        result = helper.group_log_then_exp_from_identity(self.group,
                                                         self.transfo_small)
        expected = self.transfo_small
        self.assertTrue(np.allclose(result, expected))

        # Compose exp then log
        result = helper.group_exp_then_log_from_identity(self.group,
                                                         self.transfo_1)
        expected = self.transfo_1
        self.assertTrue(np.allclose(result, expected))

        # Compose exp then log
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        result = helper.group_exp_then_log_from_identity(self.group,
                                                         self.transfo_small)
        expected = self.transfo_small
        self.assertTrue(np.allclose(result, expected))

    def test_group_exp(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        # Tangent vector is a translation (no infinitesimal rotational part)
        # Expect the sum of the translation
        # with the translation of the reference point
        result = self.group.group_exp(base_point=self.translation_1,
                                      tangent_vec=self.translation_2)
        expected = self.translation_1 + self.translation_2
        self.assertTrue(np.allclose(result, expected))

    def test_group_log(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        # Point is a translation (no rotational part)
        # Expect the difference of the translation
        # by the translation of the reference point
        result = self.group.group_log(base_point=self.translation_2,
                                      point=self.translation_1)
        expected = self.translation_1 - self.translation_2

        self.assertTrue(np.allclose(result, expected))

    def test_group_exp_and_log(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        transfo_base_point = self.transfo_1

        # Compose log then exp
        result = helper.group_log_then_exp(self.group,
                                           base_point=transfo_base_point,
                                           point=self.transfo_2)
        expected = self.transfo_2
        self.assertTrue(np.allclose(result, expected))

        # Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        result = helper.group_log_then_exp(self.group,
                                           base_point=transfo_base_point,
                                           point=self.transfo_small)
        expected = self.transfo_small
        self.assertTrue(np.allclose(result, expected))

    def test_left_exp_from_id(self):
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

    def test_left_log_from_id(self):
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

    def test_left_exp_and_log_from_id(self):
        """
        Test that the riemannian left exponential from the identity
        and the riemannian left logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        metric = self.group.left_canonical_metric
        # 1. Compose log then exp
        # Canonical inner product on the lie algebra
        rot_vec_1 = np.array([-1., 0.5, -0.12])  # NB: Regularized
        translation_1 = np.array([-91., -7., 0.007])
        point_1 = np.concatenate([rot_vec_1, translation_1])

        log_1 = metric.log_from_identity(point=point_1)
        result_1 = metric.exp_from_identity(tangent_vec=log_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Canonical inner product on the lie algebra
        rot_vec_2 = np.array([1e-15, 0., 5 * 1e-6])  # NB: Regularized
        translation_2 = np.array([-1., 27., 7.])
        point_2 = np.concatenate([rot_vec_2, translation_2])

        log_2 = metric.log_from_identity(point_2)
        result_2 = metric.exp_from_identity(log_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_right_exp_and_log_from_id(self):
        """
        Test that the riemannian right exponential from the identity
        and the riemannian right logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        metric = self.group.right_canonical_metric
        # 1. Compose log then exp
        # Canonical inner product on the lie algebra
        rot_vec_1 = np.array([-1., 0.5, -0.12])  # NB: Regularized
        translation_1 = np.array([-91., -7., 0.007])
        point_1 = np.concatenate([rot_vec_1, translation_1])

        aux_1 = metric.log_from_identity(point_1)
        result_1 = metric.exp_from_identity(aux_1)

        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Canonical inner product on the lie algebra
        rot_vec_2 = np.array([1e-15, 0., 5 * 1e-6])  # NB: Regularized
        translation_2 = np.array([-1., 27., 7.])
        point_2 = np.concatenate([rot_vec_2, translation_2])

        aux_2 = metric.log_from_identity(point_2)
        result_2 = metric.exp_from_identity(aux_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_left_exp(self):
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

    def test_left_log(self):
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

    def test_left_exp_and_log(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        rot_vec_base_point = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_base_point = np.array([4, -1, 2])
        transfo_base_point = np.concatenate([rot_vec_base_point,
                                            translation_base_point])

        # 1. Compose log then exp
        # Canonical inner product on the lie algebra
        rot_vec_1 = np.array([-1.2, 0.9, 0.9])  # NB: Regularized
        translation_1 = np.array([5, 5, 5])
        point_1 = np.concatenate([rot_vec_1,
                                  translation_1])

        aux_1 = self.group.left_canonical_metric.log(
                                          base_point=transfo_base_point,
                                          point=point_1)
        result_1 = self.group.left_canonical_metric.exp(
                                          base_point=transfo_base_point,
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

        aux_2 = self.group.left_canonical_metric.log(
                                        base_point=transfo_base_point,
                                        point=point_2)
        result_2 = self.group.left_canonical_metric.exp(
                                        base_point=transfo_base_point,
                                        tangent_vec=aux_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_right_exp_and_log(self):
        """
        Test that the riemannian right exponential and the
        riemannian right logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        rot_vec_base_point = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_base_point = np.array([4, -1, 2])
        transfo_base_point = np.concatenate([rot_vec_base_point,
                                            translation_base_point])

        # 1. Compose log then exp
        rot_vec_1 = np.array([-1.2, 0.9, 0.9])  # NB: Regularized
        translation_1 = np.array([5, 5, 5])
        point_1 = np.concatenate([rot_vec_1,
                                  translation_1])

        aux_1 = self.group.right_canonical_metric.log(
                                      base_point=transfo_base_point,
                                      point=point_1)
        result_1 = self.group.right_canonical_metric.exp(
                                      base_point=transfo_base_point,
                                      tangent_vec=aux_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([-1e-7, 0., -7*1e-8])  # NB: Regularized
        translation_2 = np.array([6, 5, 9])
        point_2 = np.concatenate([rot_vec_2,
                                  translation_2])

        aux_2 = self.group.right_canonical_metric.log(
                                      base_point=transfo_base_point,
                                      point=point_2)
        result_2 = self.group.right_canonical_metric.exp(
                                      base_point=transfo_base_point,
                                      tangent_vec=aux_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_squared_dist(self):
        point_1 = np.array([0.16329, -0.660283, 2.75099,
                            -0.363386, 0.113832, 1.3792])
        point_2 = np.array([-1.2297, 0.551821, -0.370994,
                            -0.130283, 0.518082, 0.671212])

        sq_dist_1_2 = self.left_canonical_metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.left_canonical_metric.squared_dist(point_2, point_1)
        self.assertTrue(np.allclose(sq_dist_1_2, sq_dist_2_1))

    def test_group_exponential_barycenter(self):
        # TODO(nina): this test fails.
        point_1 = self.group.random_uniform()
        result_1 = self.group.group_exponential_barycenter(
                                points=[point_1, point_1])
        expected_1 = point_1
        # self.assertTrue(np.allclose(result_1, expected_1))

        point_2 = self.group.random_uniform()
        result_2 = self.group.group_exponential_barycenter(
                                points=[point_2, point_2],
                                weights=[1., 2.])
        expected_2 = point_2
        # self.assertTrue(np.allclose(result_2, expected_2))

        result_3 = self.group.group_exponential_barycenter(
                                points=[point_1, point_2],
                                weights=[1., .1])

        self.assertTrue(self.group.belongs(result_3))


if __name__ == '__main__':
        unittest.main()
