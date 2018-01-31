"""Unit tests for losses."""

import numpy as np
import unittest
import logging

import geomstats.losses as losses

from geomstats.special_euclidean_group import SpecialEuclideanGroup

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SE3_GROUP = SpecialEuclideanGroup(n=3)
LEFT_CANONICAL_METRIC = SE3_GROUP.left_canonical_metric


def test_partial_numerical_gradient_check(dx, xdim, rnd=False):

    logger.info('Testing dL/dy_pred[%d]', xdim)

    if rnd is False:
        # Use predefined regularized transformation vector
        logger.debug('Using predefined regularized transformation vector')
        y_pred = np.array([-0.662919, -1.29015, 1.30245,
                           1.25567, 1.09228, 0.153399])
        y_true = np.array([0.0349858, 0.562573, -1.94181,
                           -0.173889, 0.127527, -0.19467])
        logger.debug('y_pred: %s', str(y_pred))
        logger.debug('y_true: %s', str(y_true))
    else:
        # Use random regularized transformation vector
        logger.debug('Using random regularized transformation vector')
        y_pred = SE3_GROUP.random_uniform()
        y_true = SE3_GROUP.random_uniform()
        logger.debug('y_pred: %s', str(y_pred))
        logger.debug('y_true: %s', str(y_true))

    forward_loss = losses.lie_group_riemannian_loss(y_pred, y_true)
    backward_grad = losses.lie_group_riemannian_grad(y_pred, y_true)
    logger.debug('Forward Loss:  %f', forward_loss)
    logger.debug('Backward Grad:  %s', str(backward_grad))

    dx_vec = np.zeros(6)
    dx_vec[xdim] = dx

    y_pred_plus_dx = y_pred + dx_vec
    y_pred_minus_dx = y_pred - dx_vec
    logger.debug('y_pred[%d]+dx:  %s', xdim, str(y_pred_plus_dx))
    logger.debug('y_pred[%d]-dx:  %s', xdim, str(y_pred_minus_dx))

    forward_loss_y_pred_plus_dx = losses.lie_group_riemannian_loss(
                                                        y_pred_plus_dx,
                                                        y_true)
    forward_loss_y_pred_minus_dx = losses.lie_group_riemannian_loss(
                                                        y_pred_minus_dx,
                                                        y_true)
    logger.debug('Forward with +dx:  %s', str(forward_loss_y_pred_plus_dx))
    logger.debug('Forward with -dx:  %s', str(forward_loss_y_pred_minus_dx))

    dy_dx = ((forward_loss_y_pred_plus_dx - forward_loss_y_pred_minus_dx)
             / (dx * 2))
    logger.debug('Computed grad backward_grad[%d]:  %f',
                 xdim, backward_grad[xdim])
    logger.debug('Calculated grad dL/dy_pred[%d]:  %f',
                 xdim, dy_dx)

    grad_diff = np.abs(backward_grad[xdim]-dy_dx)
    scale = np.max(np.fabs(np.array([backward_grad[xdim], dy_dx, 1.])))

    return grad_diff, scale


class TestLossesMethods(unittest.TestCase):
    def test_loss_and_grad_1(self):
        # Both regularized
        # Test 1st component
        rot_vec_1 = np.array([-1.2, .1, .97])  # NB: Regularized
        translation_1 = np.array([56, -5, 29])
        y_true_1 = np.concatenate([rot_vec_1,
                                   translation_1])
        rot_vec_1 = np.array([1.4, -.9, -.9])  # NB: Regularized
        translation_1 = np.array([3, 6, 1])
        y_pred_1 = np.concatenate([rot_vec_1,
                                   translation_1])

        expected_1 = losses.lie_group_riemannian_numerical_grad_per_coord(
                                                              y_pred_1,
                                                              y_true_1)

        result_1 = losses.lie_group_riemannian_grad(y_pred_1,
                                                    y_true_1)

        # print('\nLinear numerical gradient:')
        # print(expected_1)

        # print('\nGradient from formula:')
        # print(result_1[0])

        # self.assertTrue(np.allclose(result_1, expected_1))

    def test_loss_and_grad_2(self):
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

        expected_2 = losses.lie_group_riemannian_numerical_grad_per_coord(
                                                              y_pred_2,
                                                              y_true_2)

        result_2 = losses.lie_group_riemannian_grad(y_pred_2,
                                                    y_true_2)

        # print('\nLinear numerical gradient:')
        # print(expected_2)

        # print('\nGradient from formula:')
        # print(result_2[0])

        # self.assertTrue(np.allclose(result_2, expected_2))

    def test_numerical_gradient_check_0(self):

        stepsize = 0.01
        threshold = 0.01
        grad_diff, scale = test_partial_numerical_gradient_check(stepsize,
                                                                 0,
                                                                 True)
        self.assertLess(grad_diff, scale * threshold, msg='FAILURE: dL/dy_pred[0]')

    def test_numerical_gradient_check_1(self):

        stepsize = 0.01
        threshold = 0.01
        grad_diff, scale = test_partial_numerical_gradient_check(stepsize, 1, True)
        self.assertLess(grad_diff, scale * threshold, msg='FAILURE: dL/dy_pred[1]')

    def test_numerical_gradient_check_2(self):

        stepsize = 0.01
        threshold = 0.01
        grad_diff, scale = test_partial_numerical_gradient_check(stepsize, 2, True)
        self.assertLess(grad_diff, scale * threshold, msg='FAILURE: dL/dy_pred[2]')

    def test_numerical_gradient_check_3(self):

        stepsize = 0.01
        threshold = 0.01
        grad_diff, scale = test_partial_numerical_gradient_check(stepsize, 3, True)
        self.assertLess(grad_diff, scale * threshold, msg='FAILURE: dL/dy_pred[3]')

    def test_numerical_gradient_check_4(self):

        stepsize = 0.01
        threshold = 0.01
        grad_diff, scale = test_partial_numerical_gradient_check(stepsize, 4, True)
        self.assertLess(grad_diff, scale * threshold, msg='FAILURE: dL/dy_pred[4]')

    def test_numerical_gradient_check_5(self):

        stepsize = 0.01
        threshold = 0.01
        grad_diff, scale = test_partial_numerical_gradient_check(stepsize, 5, True)
        self.assertLess(grad_diff, scale * threshold, msg='FAILURE: dL/dy_pred[5]')


if __name__ == '__main__':
    unittest.main()
