"""Unit tests for rigid_transformations module."""

import geomstats.rigid_transformations as rigids

import numpy as np
import unittest


class TestRigidTransformationsMethods(unittest.TestCase):
    def test_compose(self):
        # 1. Composition by identity, on the right
        # Expect the original transformation
        transfo_1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        result_1 = rigids.compose(transfo_1, rigids.GROUP_IDENTITY)
        expected_1 = transfo_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Composition by identity, on the left
        # Expect the original transformation
        transfo_2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        result_2 = rigids.compose(rigids.GROUP_IDENTITY, transfo_2)
        expected_2 = transfo_2

        self.assertTrue(np.allclose(result_2, expected_2))

        # 3. Composition of translations (no rotational part)
        # Expect the sum of the translations
        transfo_a_3 = np.array([0., 0., 0., 0.4, 0.5, 0.6])
        transfo_b_3 = np.array([0., 0., 0., 0.5, 0.6, 0.7])

        result_3 = rigids.compose(transfo_a_3, transfo_b_3)
        expected_3 = np.array([0., 0., 0., 0.9, 1.1, 1.3])

        self.assertTrue(np.allclose(result_3, expected_3))

    def test_compose_and_inverse(self):
        # 1. Compose transformation by its inverse on the right
        # Expect the group identity
        transfo_1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        inv_transfo_1 = rigids.inverse(transfo_1)

        result_1 = rigids.compose(transfo_1, inv_transfo_1)
        expected_1 = rigids.GROUP_IDENTITY

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose transformation by its inverse on the left
        # Expect the group identity
        transfo_2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        inv_transfo_2 = rigids.inverse(transfo_2)

        result_2 = rigids.compose(inv_transfo_2, transfo_2)
        expected_2 = rigids.GROUP_IDENTITY

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_group_log_from_identity(self):
        # 1. Group logarithm of a translation (no rotational part)
        # Expect the original translation
        rot_vec_1 = np.array([0, 0, 0])
        translation_1 = np.array([1, 0, -3])
        transfo_1 = np.concatenate([rot_vec_1, translation_1])

        result_1 = rigids.group_log(transfo_1)
        expected_1 = transfo_1

        self.assertTrue(np.allclose(expected_1, result_1))

        # 2. Group logarithm of a transformation
        # where translation is parallel to rotation axis
        # Expect the original transformation
        rot_vec_2 = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_2 = np.array([4, 0, 0])
        transfo_2 = np.concatenate([rot_vec_2, translation_2])

        result_2 = rigids.group_log(transfo_2)
        expected_2 = transfo_2

        self.assertTrue(np.allclose(expected_2, result_2))

    def test_group_exp_from_identity(self):
        # 1. Group exponential of a translation (no rotational part)
        # Expect the original translation
        rot_vec_1 = np.array([0, 0, 0])
        translation_1 = np.array([1, 0, -3])
        tangent_vec_1 = np.concatenate([rot_vec_1, translation_1])

        result_1 = rigids.group_exp(tangent_vec_1)
        expected_1 = tangent_vec_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Group exponential of a transformation
        # where translation is parallel to rotation axis
        # Expect the original transformation
        rot_vec_2 = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_2 = np.array([4, 0, 0])
        tangent_vec_2 = np.concatenate([rot_vec_2, translation_2])

        result_2 = rigids.group_exp(tangent_vec_2)
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

        result_1 = rigids.group_exp(rigids.group_log(point_1))
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([1e-10, 0., -6 * 1e-6])  # NB: Regularized
        translation_2 = np.array([-1., 27., 7.])
        point_2 = np.concatenate([rot_vec_2, translation_2])

        result_2 = rigids.group_exp(rigids.group_log(point_2))
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

        # 3. Compose exp then log
        rot_vec_3 = np.array([0.01, -1., -0.8])  # NB: Regularized
        translation_3 = np.array([10., 2., 7.])
        tangent_vec_3 = np.concatenate([rot_vec_3, translation_3])

        result_3 = rigids.group_exp(rigids.group_log(tangent_vec_3))
        expected_3 = tangent_vec_3

        self.assertTrue(np.allclose(result_3, expected_3))

        # 4. Compose exp then log
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_4 = np.array([1e-10, 0., -6 * 1e-6])  # NB: Regularized
        translation_4 = np.array([-1., 27., 7.])
        tangent_vec_4 = np.concatenate([rot_vec_4, translation_4])

        result_4 = rigids.group_exp(rigids.group_log(tangent_vec_4))
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

        result_1 = rigids.group_exp(tangent_vec_1, ref_point=transfo_ref_point)
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

        result_1 = rigids.group_log(point_1, ref_point=transfo_ref_point)

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

        aux_1 = rigids.group_log(point_1,
                                 ref_point=transfo_ref_point)
        result_1 = rigids.group_exp(aux_1,
                                    ref_point=transfo_ref_point)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([-1e-7, 0., -7 * 1e-8])  # NB: Regularized
        translation_2 = np.array([6, 5, 9])
        point_2 = np.concatenate([rot_vec_2,
                                  translation_2])

        aux_2 = rigids.group_log(point_2,
                                 ref_point=transfo_ref_point)
        result_2 = rigids.group_exp(aux_2,
                                    ref_point=transfo_ref_point)
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

        result_1 = rigids.riemannian_exp(tangent_vec_1)
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
        result_1 = rigids.riemannian_log(transfo_1)

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([1e-8, 0., 1e-9])  # NB: Regularized
        translation_2 = np.array([10000, -5.9, -93])
        transfo_2 = np.concatenate([rot_vec_2, translation_2])

        expected_2 = transfo_2
        result_2 = rigids.riemannian_log(transfo_2)

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

        result_1 = rigids.riemannian_exp(rigids.riemannian_log(point_1))
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Canonical inner product on the lie algebra
        rot_vec_2 = np.array([1e-15, 0., 5 * 1e-6])  # NB: Regularized
        translation_2 = np.array([-1., 27., 7.])
        point_2 = np.concatenate([rot_vec_2, translation_2])

        result_2 = rigids.riemannian_exp(rigids.riemannian_log(point_2))
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

        # 3. Compose log then exp
        # Block diagonal inner product
        rot_vec_3 = np.array([-1., 0.5, -0.12])  # NB: Regularized
        translation_3 = np.array([-91., -7., 0.007])
        point_3 = np.concatenate([rot_vec_3, translation_3])

        inner_product = np.zeros([6, 6])
        inner_product[0:3, 0:3] = 3 * np.eye(3)
        inner_product[3:6, 3:6] = 9 * np.eye(3)

        aux_3 = rigids.riemannian_log(point_3,
                                      inner_product=inner_product)
        result_3 = rigids.riemannian_exp(aux_3,
                                         inner_product=inner_product)
        expected_3 = point_3

        self.assertTrue(np.allclose(result_3, expected_3))

        # 4. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Block diagonal inner product
        rot_vec_4 = np.array([1e-15, 0., 5 * 1e-6])  # NB: Regularized
        translation_4 = np.array([-1., 27., 7.])
        point_4 = np.concatenate([rot_vec_4, translation_4])

        inner_product = np.zeros([6, 6])
        inner_product[0:3, 0:3] = 3 * np.eye(3)
        inner_product[3:6, 3:6] = 9 * np.eye(3)

        aux_4 = rigids.riemannian_log(point_4,
                                      inner_product=inner_product)
        result_4 = rigids.riemannian_exp(aux_4,
                                         inner_product=inner_product)
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

        aux_1 = rigids.riemannian_log(point_1,
                                      left_or_right='right')
        result_1 = rigids.riemannian_exp(aux_1,
                                         left_or_right='right')

        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Canonical inner product on the lie algebra
        rot_vec_2 = np.array([1e-15, 0., 5 * 1e-6])  # NB: Regularized
        translation_2 = np.array([-1., 27., 7.])
        point_2 = np.concatenate([rot_vec_2, translation_2])

        aux_2 = rigids.riemannian_log(point_2,
                                      left_or_right='right')
        result_2 = rigids.riemannian_exp(aux_2,
                                         left_or_right='right')
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

        # 3. Compose log then exp
        # Block diagonal inner product
        rot_vec_3 = np.array([-1., 0.5, -0.12])  # NB: Regularized
        translation_3 = np.array([-91., -7., 0.007])
        point_3 = np.concatenate([rot_vec_3, translation_3])

        inner_product = np.zeros([6, 6])
        inner_product[0:3, 0:3] = 3 * np.eye(3)
        inner_product[3:6, 3:6] = 9 * np.eye(3)

        aux_3 = rigids.riemannian_log(point_3,
                                      inner_product=inner_product,
                                      left_or_right='right')
        result_3 = rigids.riemannian_exp(aux_3,
                                         inner_product=inner_product,
                                         left_or_right='right')
        expected_3 = point_3

        self.assertTrue(np.allclose(result_3, expected_3))

        # 4. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Block diagonal inner product
        rot_vec_4 = np.array([1e-15, 0., 5 * 1e-6])  # NB: Regularized
        translation_4 = np.array([-1., 27., 7.])
        point_4 = np.concatenate([rot_vec_4, translation_4])

        inner_product = np.zeros([6, 6])
        inner_product[0:3, 0:3] = 3 * np.eye(3)
        inner_product[3:6, 3:6] = 9 * np.eye(3)

        aux_4 = rigids.riemannian_log(point_4,
                                      inner_product=inner_product,
                                      left_or_right='right')
        result_4 = rigids.riemannian_exp(aux_4,
                                         inner_product=inner_product,
                                         left_or_right='right')
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

        result_1 = rigids.riemannian_exp(tangent_vec_1,
                                         ref_point=transfo_ref_point)
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

        result_1 = rigids.riemannian_log(point_1, ref_point=transfo_ref_point)

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

        aux_1 = rigids.riemannian_log(point_1,
                                      ref_point=transfo_ref_point)
        result_1 = rigids.riemannian_exp(aux_1,
                                         ref_point=transfo_ref_point)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        # Canonical inner product on the lie algebra
        rot_vec_2 = np.array([-1e-7, 0., -7*1e-8])  # NB: Regularized
        translation_2 = np.array([6, 5, 9])
        point_2 = np.concatenate([rot_vec_2,
                                  translation_2])

        aux_2 = rigids.riemannian_log(point_2,
                                      ref_point=transfo_ref_point)
        result_2 = rigids.riemannian_exp(aux_2,
                                         ref_point=transfo_ref_point)
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

        aux_3 = rigids.riemannian_log(point_3,
                                      ref_point=transfo_ref_point,
                                      inner_product=inner_product)
        result_3 = rigids.riemannian_exp(aux_3,
                                         ref_point=transfo_ref_point,
                                         inner_product=inner_product)
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

        aux_4 = rigids.riemannian_log(point_4,
                                      ref_point=transfo_ref_point,
                                      inner_product=inner_product)
        result_4 = rigids.riemannian_exp(aux_4,
                                         ref_point=transfo_ref_point,
                                         inner_product=inner_product)
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

        aux_1 = rigids.riemannian_log(point_1,
                                      ref_point=transfo_ref_point,
                                      left_or_right='right')
        result_1 = rigids.riemannian_exp(aux_1,
                                         ref_point=transfo_ref_point,
                                         left_or_right='right')
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose log then exp
        # for edge case: angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([-1e-7, 0., -7*1e-8])  # NB: Regularized
        translation_2 = np.array([6, 5, 9])
        point_2 = np.concatenate([rot_vec_2,
                                  translation_2])

        aux_2 = rigids.riemannian_log(point_2,
                                      ref_point=transfo_ref_point,
                                      left_or_right='right')
        result_2 = rigids.riemannian_exp(aux_2,
                                         ref_point=transfo_ref_point,
                                         left_or_right='right')
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

        aux_3 = rigids.riemannian_log(point_3,
                                      ref_point=transfo_ref_point,
                                      inner_product=inner_product,
                                      left_or_right='right')
        result_3 = rigids.riemannian_exp(aux_3,
                                         ref_point=transfo_ref_point,
                                         inner_product=inner_product,
                                         left_or_right='right')
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

        aux_4 = rigids.riemannian_log(point_4,
                                      ref_point=transfo_ref_point,
                                      inner_product=inner_product,
                                      left_or_right='right')
        result_4 = rigids.riemannian_exp(aux_4,
                                         ref_point=transfo_ref_point,
                                         inner_product=inner_product,
                                         left_or_right='right')
        expected_4 = point_4

        self.assertTrue(np.allclose(result_4, expected_4))

    def test_riemannian_dist(self):
        # Both regularized
        rot_vec_1 = np.array([-1., 0., -.7])  # NB: Regularized
        translation_1 = np.array([6, 5, 9])
        point_a_1 = np.concatenate([rot_vec_1,
                                    translation_1])
        rot_vec_1 = np.array([-1.2, 0.9, 0.9])  # NB: Regularized
        translation_1 = np.array([5, 5, 5])
        point_b_1 = np.concatenate([rot_vec_1,
                                    translation_1])

        dist_a_b_1 = rigids.riemannian_dist(point_a_1, point_b_1)
        dist_b_a_1 = rigids.riemannian_dist(point_b_1, point_a_1)

        self.assertTrue(dist_a_b_1, dist_b_a_1)

        # point_a not regularized, point_b regularized
        rot_vec_2 = np.array([-10., 0., -4.7])  # NB: Regularized
        translation_2 = np.array([6, 5, 9])
        point_a_2 = np.concatenate([rot_vec_2,
                                    translation_2])
        rot_vec_2 = np.array([-1.2, 0.9, 0.9])  # NB: Regularized
        translation_2 = np.array([5, 5, 5])
        point_b_2 = np.concatenate([rot_vec_2,
                                    translation_2])

        dist_a_b_2 = rigids.riemannian_dist(point_a_2, point_b_2)
        dist_b_a_2 = rigids.riemannian_dist(point_b_2, point_a_2)

        self.assertTrue(dist_a_b_2, dist_b_a_2)


if __name__ == '__main__':
        unittest.main()
