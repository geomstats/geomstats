"""Unit tests for losses."""

import numpy as np
import unittest
import logging

import geomstats.losses as losses

from geomstats.special_euclidean_group import SpecialEuclideanGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SE3_GROUP = SpecialEuclideanGroup(n=3)
SO3_GROUP = SpecialOrthogonalGroup(n=3)
LEFT_CANONICAL_METRIC = SE3_GROUP.left_canonical_metric


def test_partial_numerical_gradient_check(dx, xdim, rnd=False):

    logger.info('Testing dL/dy_pred[%d]', xdim)

    if rnd is False:
        # Use predefined regularized transformation vector
        logger.debug('Using predefined regularized transformation vector')
        y_pred = np.array([1.37186, -1.89342, 0.602668,
                          -0.588076, -0.16529, -0.235406])

        y_true = np.array([-1.6099, -0.537204, -0.293682,
                           -0.125601, 2.1101, -0.238697])

        logger.debug('y_pred: %s', str(y_pred))
        logger.debug('y_true: %s', str(y_true))
    else:
        # Use random regularized transformation vector
        logger.debug('Using random regularized transformation vector')
        y_pred = SE3_GROUP.random_uniform()
        y_true = SE3_GROUP.random_uniform()
        logger.debug('y_pred: %s', str(y_pred))
        logger.debug('y_true: %s', str(y_true))

    rot_vec_pred = y_pred[:3]
    rot_vec_true = y_true[:3]
    logger.debug('Norm of the rotation vector of y_pred: %s', str(np.linalg.norm(rot_vec_pred)))
    logger.debug('Norm of the rotation vector of y_true: %s', str(np.linalg.norm(rot_vec_true)))


    dist = SO3_GROUP.bi_invariant_metric.dist(rot_vec_pred, rot_vec_true)
    logger.debug('Distance between rotation vectors: %s', str(dist))
    logger.debug('Difference with pi: %s', str(dist - np.pi))

    metric = LEFT_CANONICAL_METRIC.inner_product_matrix(base_point=y_pred)
    metric_rot = metric[:3, :3]
    logger.debug('Metric matrix rotations at y_pred:\n %s', str(metric_rot))
    metric_trans = metric[3:6, 3:6]
    logger.debug('Metric matrix translations at y_pred:\n %s', str(metric_trans))

    forward_loss = losses.lie_group_riemannian_loss(y_pred, y_true)
    backward_grad = losses.lie_group_riemannian_grad(y_pred, y_true)
    logger.debug('Forward Loss:  %f', forward_loss)
    logger.debug('Backward Grad:  %s', str(backward_grad))

    norm = SO3_GROUP.bi_invariant_metric.norm(backward_grad[:3],
                                              base_point=rot_vec_pred)
    logger.debug('Norm of half of the rotation part of the Backward Grad: %s', str(norm / 2))
    dx_vec = np.zeros(6)
    dx_vec[xdim] = dx

    y_pred_plus_dx = y_pred + dx_vec
    y_pred_minus_dx = y_pred - dx_vec
    logger.debug('y_pred[%d]+dx:  %s', xdim, str(y_pred_plus_dx))
    logger.debug('y_pred[%d]-dx:  %s', xdim, str(y_pred_minus_dx))

    rot_vec_pred_plus_dx = y_pred_plus_dx[:3]
    rot_vec_true = y_true[:3]
    dist = SO3_GROUP.bi_invariant_metric.dist(rot_vec_pred_plus_dx, rot_vec_true)
    logger.debug('Distance between rotation vectors, plus: %s', str(dist))
    logger.debug('Difference with pi: %s', str(dist - np.pi))

    rot_vec_pred_minus_dx = y_pred_minus_dx[:3]
    rot_vec_true = y_true[:3]
    dist = SO3_GROUP.bi_invariant_metric.dist(rot_vec_pred_minus_dx, rot_vec_true)
    logger.debug('Distance between rotation vectors, minus: %s', str(dist))
    logger.debug('Difference with pi: %s', str(dist - np.pi))

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
    def test_numerical_gradient_check_0(self):

        stepsize = 0.01
        threshold = 0.01
        grad_diff, scale = test_partial_numerical_gradient_check(
                                                 stepsize, 0, False)
        self.assertLess(grad_diff, scale * threshold,
                        msg='FAILURE: dL/dy_pred[0]')

    def test_numerical_gradient_check_1(self):

        stepsize = 0.01
        threshold = 0.01
        grad_diff, scale = test_partial_numerical_gradient_check(
                                                 stepsize, 1, False)
        self.assertLess(grad_diff, scale * threshold,
                        msg='FAILURE: dL/dy_pred[1]')

    def test_numerical_gradient_check_2(self):

        stepsize = 0.01
        threshold = 0.01
        grad_diff, scale = test_partial_numerical_gradient_check(
                                                 stepsize, 2, False)
        self.assertLess(grad_diff, scale * threshold,
                        msg='FAILURE: dL/dy_pred[2]')

    def test_numerical_gradient_check_3(self):

        stepsize = 0.01
        threshold = 0.01
        grad_diff, scale = test_partial_numerical_gradient_check(
                                                 stepsize, 3, False)
        self.assertLess(grad_diff, scale * threshold,
                        msg='FAILURE: dL/dy_pred[3]')

    def test_numerical_gradient_check_4(self):

        stepsize = 0.01
        threshold = 0.01
        grad_diff, scale = test_partial_numerical_gradient_check(
                                                 stepsize, 4, False)
        self.assertLess(grad_diff, scale * threshold,
                        msg='FAILURE: dL/dy_pred[4]')

    def test_numerical_gradient_check_5(self):

        stepsize = 0.01
        threshold = 0.01
        grad_diff, scale = test_partial_numerical_gradient_check(
                                                 stepsize, 5, False)
        self.assertLess(grad_diff, scale * threshold,
                        msg='FAILURE: dL/dy_pred[5]')


if __name__ == '__main__':
    unittest.main()
