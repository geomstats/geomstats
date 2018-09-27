"""
Unit tests for General Linear group.
"""

import importlib
import math
import numpy as np
import os
import tensorflow as tf

import geomstats.backend as gs
import tests.helper as helper

from geomstats.general_linear_group import GeneralLinearGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

RTOL = 1e-5


class TestGeneralLinearGroupTensorFlow(tf.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        n = 3
        self.group = GeneralLinearGroup(n=n)
        # We generate invertible matrices using so3_group
        self.so3_group = SpecialOrthogonalGroup(n=n)
        
        
        
    @classmethod
    def setUpClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
        importlib.reload(gs)

    @classmethod
    def tearDownClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        importlib.reload(gs)


    def test_belongs(self):
        """
        A rotation matrix belongs to the matrix Lie group
        of invertible matrices.
        """
        rot_vec = self.so3_group.random_uniform()
        rot_mat = self.so3_group.matrix_from_rotation_vector(rot_vec)
        expected = tf.convert_to_tensor([True])

        with self.test_session():
             print(gs.eval(self.group.belongs(rot_mat)))
             self.assertAllClose(gs.eval(self.group.belongs(rot_mat)),
                                 gs.eval(excpected))
# 
#     def test_compose(self):
#         # 1. Composition by identity, on the right
#         # Expect the original transformation
#         rot_vec_1 = self.so3_group.random_uniform()
#         mat_1 = self.so3_group.matrix_from_rotation_vector(rot_vec_1)
# 
#         result_1 = self.group.compose(mat_1, self.group.identity)
#         expected_1 = mat_1
# 
#         with self.test_session():
#             self.assertAllClose(gs.eval(result_1), gs.eval(expected_1))
# 
#         # 2. Composition by identity, on the left
#         # Expect the original transformation
#         rot_vec_2 = self.so3_group.random_uniform()
#         mat_2 = self.so3_group.matrix_from_rotation_vector(rot_vec_2)
# 
#         result_2 = self.group.compose(self.group.identity, mat_2)
#         expected_2 = mat_2
# 
#         norm = gs.linalg.norm(expected_2)
#         atol = RTOL
#         if norm != 0:
#             atol = RTOL * norm
#         with self.test_session():
#             self.assertAllClose(gs.eval(result_2), gs.eval(expected_2), atol=atol)
# 
#     def test_compose_and_inverse(self):
#         # 1. Compose transformation by its inverse on the right
#         # Expect the group identity
#         rot_vec_1 = self.so3_group.random_uniform()
#         mat_1 = self.so3_group.matrix_from_rotation_vector(rot_vec_1)
#         inv_mat_1 = self.group.inverse(mat_1)
# 
#         result_1 = self.group.compose(mat_1, inv_mat_1)
#         expected_1 = self.group.identity
# 
#         norm = gs.linalg.norm(expected_1)
#         atol = RTOL
#         if norm != 0:
#             atol = RTOL * norm
#             
#         with self.test_session():
#             self.assertAllClose(gs.eval(result_1), gs.eval(expected_1), atol=atol)
# 
#         # 2. Compose transformation by its inverse on the left
#         # Expect the group identity
#         rot_vec_2 = self.so3_group.random_uniform()
#         mat_2 = self.so3_group.matrix_from_rotation_vector(rot_vec_2)
#         inv_mat_2 = self.group.inverse(mat_2)
# 
#         result_2 = self.group.compose(inv_mat_2, mat_2)
#         expected_2 = self.group.identity
# 
#         norm = gs.linalg.norm(expected_2)
#         atol = RTOL
#         if norm != 0:
#             atol = RTOL * norm
# 
#         with self.test_session():
#             self.assertAllClose(gs.eval(result_2), gs.eval(expected_2), atol=atol)

if __name__ == '__main__':
        tf.test.main()
