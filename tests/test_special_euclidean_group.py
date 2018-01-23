"""Unit tests for special euclidean group module."""

import numpy as np
import unittest

from geomstats.special_euclidean_group import SpecialEuclideanGroup


class TestSpecialEuclideanGroupMethods(unittest.TestCase):
    N = 3
    GROUP = SpecialEuclideanGroup(n=N)

    def test_random_and_belongs(self):
        """
        Test that the random uniform method samples
        on the special euclidean group.
        """
        base_point = self.GROUP.random_uniform()
        self.assertTrue(self.GROUP.belongs(base_point))

    def test_regularize(self):
        rot_vec_0 = np.array([0., 0., 0.])
        rot_vec_0 = self.GROUP.regularize(rot_vec_0)
        rot_vec_0_expected = np.array([0., 0., 0.])
        self.assertTrue(np.allclose(rot_vec_0, rot_vec_0_expected))

        rot_vec_1 = 2.5 * np.pi * np.array([0., 0., 1.])
        rot_vec_1 = self.GROUP.regularize(rot_vec_1)
        rot_vec_1_expected = np.pi / 2. * np.array([0., 0., 1.])
        self.assertTrue(np.allclose(rot_vec_1, rot_vec_1_expected))

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

        result_1 = self.GROUP.group_log(base_point=self.GROUP.identity,
                                        point=transfo_1)
        expected_1 = transfo_1

        self.assertTrue(np.allclose(expected_1, result_1))

        # 2. Group logarithm of a transformation
        # where translation is parallel to rotation axis
        # Expect the original transformation
        rot_vec_2 = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_2 = np.array([4, 0, 0])
        transfo_2 = np.concatenate([rot_vec_2, translation_2])

        result_2 = self.GROUP.group_log(base_point=self.GROUP.identity,
                                        point=transfo_2)
        expected_2 = transfo_2

        self.assertTrue(np.allclose(expected_2, result_2))

    def test_group_exp_from_identity(self):
        # 1. Group exponential of a translation (no rotational part)
        # Expect the original translation
        rot_vec_1 = np.array([0, 0, 0])
        translation_1 = np.array([1, 0, -3])
        tangent_vec_1 = np.concatenate([rot_vec_1, translation_1])

        result_1 = self.GROUP.group_exp(base_point=self.GROUP.identity,
                                        tangent_vec=tangent_vec_1)
        expected_1 = tangent_vec_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Group exponential of a transformation
        # where translation is parallel to rotation axis
        # Expect the original transformation
        rot_vec_2 = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_2 = np.array([4, 0, 0])
        tangent_vec_2 = np.concatenate([rot_vec_2, translation_2])

        result_2 = self.GROUP.group_exp(base_point=self.GROUP.identity,
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

        aux_1 = self.GROUP.group_log(base_point=self.GROUP.identity,
                                     point=point_1)
        result_1 = self.GROUP.group_exp(base_point=self.GROUP.identity,
                                        tangent_vec=aux_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([1e-10, 0., -6 * 1e-6])  # NB: Regularized
        translation_2 = np.array([-1., 27., 7.])
        point_2 = np.concatenate([rot_vec_2, translation_2])

        aux_2 = self.GROUP.group_log(base_point=self.GROUP.identity,
                                     point=point_2)
        result_2 = self.GROUP.group_exp(base_point=self.GROUP.identity,
                                        tangent_vec=aux_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

        # 3. Compose exp then log
        rot_vec_3 = np.array([0.01, -1., -0.8])  # NB: Regularized
        translation_3 = np.array([10., 2., 7.])
        tangent_vec_3 = np.concatenate([rot_vec_3, translation_3])

        aux_3 = self.GROUP.group_exp(base_point=self.GROUP.identity,
                                     tangent_vec=tangent_vec_3)
        result_3 = self.GROUP.group_log(base_point=self.GROUP.identity,
                                        point=aux_3)
        expected_3 = tangent_vec_3

        self.assertTrue(np.allclose(result_3, expected_3))

        # 4. Compose exp then log
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_4 = np.array([1e-10, 0., -6 * 1e-6])  # NB: Regularized
        translation_4 = np.array([-1., 27., 7.])
        tangent_vec_4 = np.concatenate([rot_vec_4, translation_4])

        aux_4 = self.GROUP.group_exp(base_point=self.GROUP.identity,
                                     tangent_vec=tangent_vec_4)
        result_4 = self.GROUP.group_log(base_point=self.GROUP.identity,
                                        point=aux_4)
        expected_4 = tangent_vec_4

        self.assertTrue(np.allclose(result_4, expected_4))

    def test_group_exp(self):
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

        result_1 = self.GROUP.group_exp(base_point=transfo_base_point,
                                        tangent_vec=tangent_vec_1)
        expected_1 = np.concatenate([np.array([0., 0., 0.]),
                                     np.array([5, -1, 9997])])
        self.assertTrue(np.allclose(result_1, expected_1))

    def test_group_log(self):
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
        translation_1 = np.array([5., 8., -3.2])
        point_1 = np.concatenate([rot_vec_1,
                                  translation_1])

        expected_1 = np.concatenate([np.array([0., 0., 0.]),
                                     np.array([1., 8., -3.2])])

        result_1 = self.GROUP.group_log(base_point=transfo_base_point,
                                        point=point_1)

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_group_exp_and_log(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
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

        aux_1 = self.GROUP.group_log(base_point=transfo_base_point,
                                     point=point_1)
        result_1 = self.GROUP.group_exp(base_point=transfo_base_point,
                                        tangent_vec=aux_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([-1e-7, 0., -7 * 1e-8])  # NB: Regularized
        translation_2 = np.array([6, 5, 9])
        point_2 = np.concatenate([rot_vec_2,
                                  translation_2])

        aux_2 = self.GROUP.group_log(base_point=transfo_base_point,
                                     point=point_2)
        result_2 = self.GROUP.group_exp(base_point=transfo_base_point,
                                        tangent_vec=aux_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_left_exp_from_id(self):
        # Riemannian left-invariant metric given by
        # the canonical inner product on the lie algebra
        # Expect the identity function
        # because we use the riemannian left logarithm with canonical
        # inner product to parameterize the transformations
        metric = self.GROUP.left_canonical_metric
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

        metric = self.GROUP.left_canonical_metric
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
        metric = self.GROUP.left_canonical_metric
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
        metric = self.GROUP.right_canonical_metric
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

        result_1 = self.GROUP.left_canonical_metric.exp(
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

        result_1 = self.GROUP.left_canonical_metric.log(
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

        aux_1 = self.GROUP.left_canonical_metric.log(
                                          base_point=transfo_base_point,
                                          point=point_1)
        result_1 = self.GROUP.left_canonical_metric.exp(
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

        aux_2 = self.GROUP.left_canonical_metric.log(
                                        base_point=transfo_base_point,
                                        point=point_2)
        result_2 = self.GROUP.left_canonical_metric.exp(
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

        aux_1 = self.GROUP.right_canonical_metric.log(
                                      base_point=transfo_base_point,
                                      point=point_1)
        result_1 = self.GROUP.right_canonical_metric.exp(
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

        aux_2 = self.GROUP.right_canonical_metric.log(
                                      base_point=transfo_base_point,
                                      point=point_2)
        result_2 = self.GROUP.right_canonical_metric.exp(
                                      base_point=transfo_base_point,
                                      tangent_vec=aux_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_group_exponential_barycenter(self):
        # TODO(nina): this test fails.
        point_1 = self.GROUP.random_uniform()
        result_1 = self.GROUP.group_exponential_barycenter(
                                points=[point_1, point_1])
        expected_1 = point_1
        # self.assertTrue(np.allclose(result_1, expected_1))

        point_2 = self.GROUP.random_uniform()
        result_2 = self.GROUP.group_exponential_barycenter(
                                points=[point_2, point_2],
                                weights=[1., 2.])
        expected_2 = point_2
        # self.assertTrue(np.allclose(result_2, expected_2))

        result_3 = self.GROUP.group_exponential_barycenter(
                                points=[point_1, point_2],
                                weights=[1., .1])

        self.assertTrue(self.GROUP.belongs(result_3))


if __name__ == '__main__':
        unittest.main()
