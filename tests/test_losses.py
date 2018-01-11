"""Unit tests for losses."""

import geomstats.losses as losses

import numpy as np
import unittest


class TestLossesMethods(unittest.TestCase):
    def test_rigids_riemannian_loss_and_grad(self):
        # Both regularized
        rot_vec_1 = np.array([-1.2, .1, .97])  # NB: Regularized
        translation_1 = np.array([56, -5, 29])
        y_true_1 = np.concatenate([rot_vec_1,
                                   translation_1])
        rot_vec_1 = np.array([1.4, -.9, -.9])  # NB: Regularized
        translation_1 = np.array([3, 6, 1])
        y_pred_1 = np.concatenate([rot_vec_1,
                                   translation_1])

        delta_1 = .01
        y_true_and_delta_1 = y_true_1 + np.array([delta_1, 0., 0., 0., 0., 0.])

        loss = losses.rigids_riemannian_loss(y_pred_1,
                                             y_true_1)
        loss_delta = losses.rigids_riemannian_loss(y_pred_1,
                                                   y_true_and_delta_1)

        expected_1 = (loss_delta - loss) / delta_1
        result_1 = losses.rigids_riemannian_grad(y_true_1,
                                                 y_pred_1)

        self.assertTrue(np.allclose(result_1, expected_1))


if __name__ == '__main__':
        unittest.main()
