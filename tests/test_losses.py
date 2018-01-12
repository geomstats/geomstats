"""Unit tests for losses."""

import math
import numpy as np
import unittest

import geomstats.losses as losses
import geomstats.rigid_transformations as rigids


class TestLossesMethods(unittest.TestCase):
    def test_rigids_riemannian_loss_and_grad(self):
        # Both regularized
        # Test 1st component
        print('\n #####  TEST_1 ##### \n')
        rot_vec_1 = np.array([-1.2, .1, .97])  # NB: Regularized
        translation_1 = np.array([56, -5, 29])
        y_true_1 = np.concatenate([rot_vec_1,
                                   translation_1])
        rot_vec_1 = np.array([1.4, -.9, -.9])  # NB: Regularized
        translation_1 = np.array([3, 6, 1])
        y_pred_1 = np.concatenate([rot_vec_1,
                                   translation_1])

        delta_1 = np.array([0.001, 0., 0., 0., 0., 0.])
        linear_y_true_and_delta_1 = y_true_1 + delta_1
        curved_y_true_and_delta_1 = rigids.riemannian_exp(
                                             delta_1,
                                             ref_point=y_true_1)

        loss_1 = losses.rigids_riemannian_loss(y_pred_1,
                                               y_true_1)

        linear_loss_delta_1 = losses.rigids_riemannian_loss(
                                             y_pred_1,
                                             linear_y_true_and_delta_1)
        curved_loss_delta_1 = losses.rigids_riemannian_loss(
                                             y_pred_1,
                                             curved_y_true_and_delta_1)

        linear_norm_delta_1 = np.linalg.norm(delta_1)
        curved_norm_delta_1 = math.sqrt(rigids.square_riemannian_norm(
                                             delta_1,
                                             ref_point=y_true_1))
        linear_expected_1 = ((linear_loss_delta_1 - loss_1)
                             / linear_norm_delta_1)
        curved_expected_1 = ((curved_loss_delta_1 - loss_1)
                             / curved_norm_delta_1)
        curved_bis_expected_1 = ((curved_loss_delta_1 - loss_1)
                                 / linear_norm_delta_1)

        result_1 = losses.rigids_riemannian_grad(y_true_1,
                                                 y_pred_1)[0]

        print('Linear numerical gradient:')
        print(linear_expected_1)
        print('\nCurved numerical gradient:')
        print(curved_expected_1)
        print('Curved bis numerical gradient:')
        print(curved_bis_expected_1)
        print('\nGradient from formula:')
        print(result_1)

        # self.assertTrue(np.allclose(result_1, curved_expected_1))

        print('\n #####  TEST_2 ##### \n')
        # Both regularized
        # Test 1st component
        rot_vec_2 = np.array([-0.2, 1., -.7])  # NB: Regularized
        translation_2 = np.array([6., -15., 9.])
        y_true_2 = np.concatenate([rot_vec_2,
                                   translation_2])
        rot_vec_2 = np.array([-1.2, -1.9, 0.])  # NB: Regularized
        translation_2 = np.array([31, .6, -11])
        y_pred_2 = np.concatenate([rot_vec_2,
                                   translation_2])

        delta_2 = np.array([0.001, 0., 0., 0., 0., 0.])
        linear_y_true_and_delta_2 = y_true_2 + delta_2
        curved_y_true_and_delta_2 = rigids.riemannian_exp(
                                             delta_2,
                                             ref_point=y_true_2)

        loss_2 = losses.rigids_riemannian_loss(y_pred_2,
                                               y_true_2)

        linear_loss_delta_2 = losses.rigids_riemannian_loss(
                                             y_pred_2,
                                             linear_y_true_and_delta_2)
        curved_loss_delta_2 = losses.rigids_riemannian_loss(
                                             y_pred_2,
                                             curved_y_true_and_delta_2)

        linear_norm_delta_2 = np.linalg.norm(delta_2)
        curved_norm_delta_2 = math.sqrt(rigids.square_riemannian_norm(
                                             delta_2,
                                             ref_point=y_true_2))
        linear_expected_2 = ((linear_loss_delta_2 - loss_2)
                             / linear_norm_delta_2)
        curved_expected_2 = ((curved_loss_delta_2 - loss_2)
                             / curved_norm_delta_2)
        curved_bis_expected_2 = ((curved_loss_delta_2 - loss_2)
                                 / linear_norm_delta_2)

        result_2 = losses.rigids_riemannian_grad(y_true_2,
                                                 y_pred_2)[0]

        print('Linear numerical gradient:')
        print(linear_expected_2)
        print('\nCurved numerical gradient:')
        print(curved_expected_2)
        print('Curved bis numerical gradient:')
        print(curved_bis_expected_2)
        print('\nGradient from formula:')
        print(result_2)

        # self.assertTrue(np.allclose(result_2, curved_expected_2))


if __name__ == '__main__':
        unittest.main()
