"""Unit tests for rigid_transformations module."""

import numpy as np
import unittest

import rigid_transformations as rigids


class TestRigidTransformationsMethods(unittest.TestCase):

    def test_rigids_group_exp_log(self):
        translation_1 = np.array([1, 0, -3])
        rot_vec_1 = np.pi / (3 * np.sqrt(2)) * np.array([0, 0, 0])
        transfo_1 = np.concatenate([rot_vec_1, translation_1])

        translation_2 = np.array([4, 0, 0])
        rot_vec_2 = np.pi / (2 * np.sqrt(3)) * np.array([1, 0, 0])
        transfo_2 = np.concatenate([rot_vec_2, translation_2])

        translation_3 = np.array([1.2, -3.6, 50])
        rot_vec_3 = np.pi / (2 * np.sqrt(3)) * np.array([1, -20, 50])
        transfo_3 = np.concatenate([rot_vec_3, translation_3])

        translation_4 = np.array([4, 10, -2])
        rot_vec_4 = (np.pi / (2 * np.sqrt(3)) *
                     np.array([6 * 1e-8, 5.5 * 1e-7, -2 * 1e-6]))
        transfo_4 = np.concatenate([rot_vec_4, translation_4])

        all_transfos = [transfo_1, transfo_2, transfo_3, transfo_4]
        for transfo in all_transfos:
            gp_log = rigids.group_log(transfo)
            transfo_result = rigids.group_exp(gp_log)
            transfo_expected = transfo

            self.assertTrue(np.allclose(transfo_result, transfo_expected))

    def test_rigids_riemannian_exp_log(self):
        # TODO(nina): This test does not pass. Find the bug.
        translation_ref_point = np.array([1, 2, 3])
        rot_vec_ref_point = np.array([-1, 3, 6])
        transfo_ref_point = np.concatenate([rot_vec_ref_point,
                                            translation_ref_point])

        translation_1 = np.array([1, 0, -3])
        rot_vec_1 = np.pi / (3 * np.sqrt(2)) * np.array([0, 1, 0])
        transfo_1 = np.concatenate([rot_vec_1, translation_1])

        translation_2 = np.array([4, 0, 0])
        rot_vec_2 = np.pi / (2 * np.sqrt(3)) * np.array([1, 0, 0])
        transfo_2 = np.concatenate([rot_vec_2, translation_2])

        translation_3 = np.array([1.2, -3.6, 50])
        rot_vec_3 = np.pi / (2 * np.sqrt(3)) * np.array([1, -20, 50])
        transfo_3 = np.concatenate([rot_vec_3, translation_3])

        translation_4 = np.array([4, 10, -2])
        rot_vec_4 = (np.pi / (2 * np.sqrt(3)) *
                     np.array([6 * 1e-8, 5.5 * 1e-7, -2 * 1e-6]))
        transfo_4 = np.concatenate([rot_vec_4, translation_4])

        all_transfos = [transfo_1, transfo_2, transfo_3, transfo_4]
        i_debug = 1
        for transfo in all_transfos:
            print('at i=%d' % i_debug)
            i_debug += 1
            riem_log = rigids.riemannian_log(
                                                 transfo,
                                                 ref_point=transfo_ref_point)
            transfo_result = rigids.riemannian_exp(
                                                 riem_log,
                                                 ref_point=transfo_ref_point)
            transfo_expected = rigids.regularize_transformation(transfo)

            self.assertTrue(np.allclose(transfo_result, transfo_expected))


if __name__ == '__main__':
        unittest.main()
