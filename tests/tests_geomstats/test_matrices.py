# import geomstats.backend as gs
# import geomstats.tests
# from geomstats.geometry.matrices import Matrices
# from geomstats.geometry.spd_matrices import SPDMatrices
# from tests.conftest import Parametrizer, TestCase, TestData

# EYE_2 = [[1.0, 0], [0.0, 1.0]]
# EYE_3 = [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
# MAT_2_3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
# MAT1_33 = [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]
# MAT2_33 = [[1.0, 2.0, 3.0], [2.0, 4.0, 7.0], [3.0, 5.0, 6.0]]
# MAT3_33 = [[0.0, 1.0, -2.0], [-1.0, 0.0, -3.0], [2.0, 3.0, 0.0]]


# class TestMatrices(TestCase, metaclass=Parametrizer):
#     class TestDataMatrices(TestData):
#         def belongs_data(self):
#             sq_mat = EYE_2
#             non_sq_mat = MAT_2_3
#             smoke_data = [
#                 dict(m=2, n=2, mat=sq_mat, expected=True),
#                 dict(m=2, n=1, mat=sq_mat, expected=False),
#                 dict(m=2, n=3, mat=non_sq_mat, expected=True),
#                 dict(m=2, n=1, mat=non_sq_mat, expected=False),
#                 dict(m=2, n=3, mat=[non_sq_mat, non_sq_mat], expected=[True, True]),
#                 dict(m=2, n=3, mat=[non_sq_mat, sq_mat], expected=[True, False]),
#             ]
#             return self.generate_tests(smoke_data)

#         def equal_data(self):

#             smoke_data = [
#                 dict(m=2, n=2, mat_1=EYE_2, mat_2=EYE_2, expected=True),
#                 dict(m=2, n=2, mat_1=EYE_2, mat_2=2 * EYE_2, expected=False),
#                 dict(
#                     m=2,
#                     n=3,
#                     mat_1=[MAT_2_3, 2 * MAT_2_3],
#                     mat_2=[MAT_2_3, 3 * MAT_2_3],
#                     expected=[True, False],
#                 ),
#             ]
#             return self.generate_tests(smoke_data)

#         def mul(self):
#             mats_1 = (
#                 [[1.0, 2.0], [3.0, 4.0]],
#                 [[-1.0, 2.0], [-3.0, 4.0]],
#                 [[1.0, -2.0], [3.0, -4.0]],
#             )
#             mats_2 = [[[2.0], [4.0]], [[1.0], [3.0]], [[1.0], [3.0]]]
#             mat_1_X_mat_2 = [[[10.0], [22.0]], [[5.0], [9.0]], [[-5.0], [-9.0]]]
#             smoke_data = [
#                 dict(mat=mats_1, expected=[[23.0, -26.0], [51.0, -58.0]]),
#                 dict(mat=(list(mats_1), mats_2), expected=mat_1_X_mat_2),
#             ]
#             return self.generate_tests(smoke_data)

#         def commutator_data(self):
#             smoke_data = []
#             return self.generate_tests(smoke_data)

#         def is_symmetric_data(self):
#             smoke_data = [
#                 dict(mat=EYE_2, expected=True),
#                 dict(mat=[EYE_2, EYE_2 + 1], expected=[True, True]),
#                 dict(mat=[MAT1_33, MAT2_33, MAT3_33], expected=[True, False, True]),
#             ]
#             self.generate_tests(smoke_data)

#         def is_skew_symmetric_data(self):
#             smoke_data = [
#                 dict(mat=EYE_2, expected=False),
#                 dict(mat=[EYE_2, -1 * EYE_2], expected=[False, False]),
#                 dict(mat=[MAT2_33, MAT3_33], expected=[False, True]),
#             ]
#             self.generate_tests(smoke_data)

#         def is_pd(self):
#             smoke_data = []
#             self.generate_tests(smoke_data)

#         def is_spd(self):
#             smoke_data = []
#             self.generate_tests(smoke_data)

#         def is_upper_triangualr(self):
#             smoke_data = [dict(mat=[])]
#             self.generate_tests(smoke_data)

#         def is_lower_triangular(self):
#             smoke_data = []
#             self.generate_tests(smoke_data)

#         def is_strictly_lower_triangular(self):
#             smoke_data = []
#             self.generate_tests(smoke_data)

#         def is_strictly_upper_triangular(self):
#             smoke_data = []
#             self.generate_tests(smoke_data)

#         def

#     def test_belongs(self, m, n, mat, expected):
#         self.assertAllClose(Matrices(m, n).belongs(gs.array(mat)), gs.array(expected))

#     def test_equal(self, m, n, mat1, mat2, expected):
#         self.assertAllClose(
#             Matrices(m, n).equal(gs.array(mat1), gs.array(mat2)), gs.array(expected)
#         )

#     def test_mul(self, mat, expected):
#         self.assertAllClose(Matrices.mul(mat), expected)

#     def test_commutator(self, mat_a, mat_b, expected):
#         self.assertAllClose(Matrices.commutator(mat_a, mat_b), expected)

#     def is_symmetric_data(self):
#         pass

#     def is_skew_symmetric_data(self):
#         smoke_data = []
#         self.generate_tests(smoke_data)

#     def is_pd(self):
#         smoke_data = []
#         self.generate_tests(smoke_data)

#     def is_spd(self):
#         smoke_data = []
#         self.generate_tests(smoke_data)

#     def is_upper_triangualr(self):
#         smoke_data = []
#         self.generate_tests(smoke_data)

#     def is_lower_triangular(self):
#         smoke_data = []
#         self.generate_tests(smoke_data)

#     def is_strictly_lower_triangular(self):
#         smoke_data = []
#         self.generate_tests(smoke_data)

#     def is_strictly_upper_triangular(self):
#         smoke_data = []
#         self.generate_tests(smoke_data)

#     def test_to_matrixtype_belongs(self, matrix_type, points):
#         self.assertAllClose(Matrices.matrix_type(points))
