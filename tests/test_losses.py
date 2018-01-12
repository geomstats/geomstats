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
                                                 y_pred_1)

        # riem_mat_1 = rigids.riemannian_metric(ref_point=y_true_1)
        # inv_riem_mat_1 = np.linalg.inv(riem_mat_1)

        jacobian_1 = rigids.jacobian_translation(y_true_1)
        inv_jacobian_1 = np.linalg.inv(jacobian_1)

        print('Linear numerical gradient:')
        print(linear_expected_1)
        print('\nCurved numerical gradient:')
        print(curved_expected_1)
        print('Curved bis numerical gradient:')
        print(curved_bis_expected_1)
        print('\nGradient from formula:')
        print(result_1[0])
        print('\nGradient from formula, translated to identity:')
        print(np.dot(inv_jacobian_1, result_1)[0])

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

    def test_caffe_gradient_checker(self):

        print('\n #####  GRADIENT CHECKER (CAFFE) ##### \n',sep='')

        print('Setting Test Parameters: ',sep='') # Adjustable Parameters
        dx = 0.01
        threshold = 0.01
        y_pred = np.array([-0.662919 , -1.29015 , 1.30245 , 1.25567 , 1.09228 , 0.153399 ]) #Regularized
        y_true = np.array([0.0349858 , 0.562573 , -1.94181 , -0.173889 , 0.127527 , -0.19467 ]) #Regularized

        print('Stepsize: ', dx,sep='')
        print('threshold: ', threshold,sep='')
        print('y_pred', y_pred,sep='')
        print('y_true', y_true,sep='')

        print('\nTesting Baseline forward and gradient: ',sep='')

        forward_loss = losses.rigids_riemannian_loss(y_pred, y_true)
        print('Forward Loss: ', forward_loss,sep='')
        
        backward_grad = losses.rigids_riemannian_grad(y_pred, y_true)
        print('Backward Grad: ', backward_grad,sep='')


        for xdim in range(0,6):
            print('\n ----- Testing dL/dy_pred[',xdim,'] ----- ',sep='')

            dx_vec = np.zeros(6)
            dx_vec[xdim] = dx

            y_pred_plus_dx = y_pred + dx_vec
            print('y_pred[',xdim,']+dx: ',y_pred_plus_dx,sep='')
            y_pred_minus_dx = y_pred - dx_vec
            print('y_pred[',xdim,']-dx: ',y_pred_minus_dx,sep='')
            
            forward_loss_y_pred_plus_dx = losses.rigids_riemannian_loss(y_pred_plus_dx, y_true)
            print('Forward with +dx: ',forward_loss_y_pred_plus_dx,sep='')
            forward_loss_y_pred_minus_dx = losses.rigids_riemannian_loss(y_pred_minus_dx, y_true)
            print('Forward with -dx: ',forward_loss_y_pred_minus_dx,sep='')
            
            dy_dx = (forward_loss_y_pred_plus_dx - forward_loss_y_pred_minus_dx) / (dx*2)
            print('Computed grad backward_grad[',xdim,']: ',backward_grad[xdim],sep='')
            print('Calculated grad dL/dy_pred[',xdim,']: ',dy_dx,sep='')
            
            diff = np.abs(backward_grad[xdim]-dy_dx)
            scale = np.max(np.fabs(np.array([backward_grad[xdim],dy_dx,1.0])))

            if diff > (scale * threshold):
                print('Gradient Check ERROR',sep='')
                print('Difference between computed and calculated: ',diff,sep='')
                print('Exceeds scale * threshold (',scale,' * ',threshold,') = ',scale*threshold,sep='')
            else:
                print('Gradient Check OK')


if __name__ == '__main__':
    unittest.main()
