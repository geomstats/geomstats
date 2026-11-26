import math

import geomstats.backend as gs
from geomstats.geometry.matrices import MatrixOperations
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.test.data import TestData

rand = gs.random.rand


class BackendTestData(TestData):
    def compose_with_inverse_test_data(self):
        smoke_data = [
            dict(
                func_name_1="linalg.logm",
                func_name_2="linalg.expm",
                a=MatrixOperations.to_diagonal(rand(3, 3)),
            ),
            dict(
                func_name_1="linalg.expm",
                func_name_2="linalg.logm",
                a=MatrixOperations.to_diagonal(rand(3, 3)),
            ),
            dict(
                func_name_1="linalg.logm",
                func_name_2="linalg.expm",
                a=SpecialOrthogonal(n=3).random_point(2),
            ),
        ]
        smoke_data += [
            dict(
                func_name_1="diag_embed",
                func_name_2="diagonal",
                a=rand(3),
                kwargs={"offset": offset},
            )
            for offset in (0, 1, -2)
        ]
        smoke_data += [
            dict(
                func_name_1=f"{tri}_embed",
                func_name_2=f"{tri}_to_vec",
                a=rand(6 if k == 0 else 3),
                kwargs={"k": k},
            )
            for tri, k in (("tril", 0), ("tril", -1), ("triu", 0), ("triu", 1))
        ]
        return self.generate_tests(smoke_data)

    def _logm_expm_data(self, func_name="linalg.logm"):
        arrays = [
            MatrixOperations.to_diagonal(rand(3, 3)),
        ]
        return [dict(func_name=func_name, a=array) for array in arrays]

    def unary_op_vec_test_data(self):
        smoke_data = [
            dict(func_name="trace", a=rand(3, 3)),
            dict(func_name="diagonal", a=rand(3, 4)),
            dict(func_name="matrix_transpose", a=rand(3, 4)),
            dict(func_name="linalg.matrix_norm", a=rand(3, 4)),
            dict(func_name="linalg.cholesky", a=SPDMatrices(3).random_point()),
            dict(func_name="linalg.eigvalsh", a=SymmetricMatrices(3).random_point()),
            dict(func_name="linalg.det", a=gs.eye(3)),
            dict(func_name="linalg.det", a=gs.array([[1.0, 2.0], [2.0, 4.0]])),
        ]

        smoke_data += self._logm_expm_data()
        smoke_data += self._logm_expm_data("linalg.expm")

        return self.generate_tests(smoke_data)

    def binary_op_vec_test_data(self):
        smoke_data = [
            dict(func_name="matmul", a=rand(3, 4), b=rand(4, 3)),
            dict(func_name="matmul", a=rand(3, 3), b=rand(3, 3)),
            dict(func_name="outer", a=rand(3), b=rand(3)),
            dict(func_name="outer", a=rand(3), b=rand(4)),
            dict(func_name="matvec", a=rand(3, 3), b=rand(3)),
            dict(func_name="matvec", a=rand(4, 3), b=rand(3)),
            dict(func_name="vecdot", a=rand(3), b=rand(3)),
            dict(func_name="cross", a=rand(3), b=rand(3)),
        ]

        return self.generate_tests(smoke_data)

    def binary_op_raises_error_test_data(self):
        smoke_data = [
            dict(func_name="matmul", a=rand(1), b=rand(1)),
            dict(func_name="matmul", a=rand(2, 3, 3), b=rand(2, 3)),
            dict(func_name="matmul", a=rand(2, 3, 3), b=rand(3, 3, 3)),
            dict(func_name="matvec", a=rand(3, 2), b=rand(3)),
            dict(func_name="vecdot", a=rand(4), b=rand(3)),
            dict(func_name="vecdot", a=rand(3, 4), b=rand(3)),
            dict(func_name="cross", a=rand(4), b=rand(4)),
        ]

        return self.generate_tests(smoke_data)

    def func_out_shape_test_data(self):
        smoke_data = []

        # take keeps the axes of size one
        smoke_data += [
            dict(func_name="take", args=(gs.ones((1, 3)), 0, -1), expected=(1,)),
            dict(func_name="take", args=(gs.ones((2, 3)), [[0, 1]], 1), expected=(2, 1, 2)),
        ]
        # without size, a random function gives a single sample
        smoke_data += [
            dict(func_name="random.normal", args=(), expected=()),
            dict(func_name="random.uniform", args=(), expected=()),
            dict(
                func_name="random.multivariate_normal",
                args=(gs.zeros(2), gs.eye(2)),
                expected=(2,),
            ),
            dict(func_name="random.choice", args=(gs.ones((4, 3)),), expected=(3,)),
            dict(func_name="random.choice", args=(gs.ones((4, 3)), 2), expected=(2, 3)),
            dict(func_name="random.choice", args=(5,), expected=()),
        ]

        return self.generate_tests(smoke_data)

    def func_out_type_test_data(self):
        smoke_data = [
            dict(func_name="shape", args=(gs.ones(3),), expected=tuple),
        ]

        return self.generate_tests(smoke_data)

    def func_out_bool_test_data(self):
        smoke_data = [
            dict(func_name="is_array", args=[gs.ones(2)], expected=True),
            dict(func_name="is_array", args=([1, 2],), expected=False),
            dict(func_name="is_array", args=(1,), expected=False),
        ]

        return self.generate_tests(smoke_data)

    def _take_data(self):
        func_name = "take"

        vec = gs.array([0, 1])
        indices = gs.array([0, 0, 1])
        # mat = gs.array(
        #     [
        #         [0, 1],
        #         [2, 3],
        #     ]
        # )
        data = [
            dict(func_name=func_name, args=[vec, indices], expected=indices),
            # TODO: uncomment after test refactor merge
            # dict(func_name=func_name, args=[mat, indices],
            #      expected=gs.array([[0, 1]] * 2 + [[2, 3]]),
            #      axis=0),
            # dict(func_name=func_name, args=[mat, indices],
            #      expected=gs.matrix_transpose(gs.array([[0, 2]] * 2 + [[1, 3]])),
            #      axis=1),
            # dict(func_name=func_name, args=[mat, 0], expected=gs.array([0, 2]),
            #      axis=1)
        ]

        return data

    def func_out_allclose_test_data(self):
        smoke_data = [
            dict(
                func_name="vecdot",
                args=(gs.array([[1.0, 2.0], [3.0, 4.0]]), gs.eye(2)),
                kwargs={"axis": 0},
                expected=gs.array([1.0, 4.0]),
            ),
            # without axis, flip reverses all axes
            dict(
                func_name="flip",
                args=(gs.array([[1.0, 2.0], [3.0, 4.0]]),),
                expected=gs.array([[4.0, 3.0], [2.0, 1.0]]),
            ),
            dict(
                func_name="linalg.matrix_rank",
                args=(gs.array([[1.0, 0.0], [0.0, 1e-3]]),),
                kwargs={"tol": 1e-2},
                expected=gs.array(1),
            ),
            # without axis, amin reduces over all axes
            dict(
                func_name="amin",
                args=(gs.array([[1.0, 2.0], [0.0, 3.0]]),),
                expected=gs.array(0.0),
            ),
            dict(
                func_name="amin",
                args=(gs.array([[1.0, 2.0], [0.0, 3.0]]),),
                kwargs={"axis": -1},
                expected=gs.array([1.0, 0.0]),
            ),
            dict(
                func_name="gamma",
                args=(gs.array([-2.5, -1.5, -0.5, 0.5]),),
                expected=math.sqrt(math.pi)
                * gs.array([-8.0 / 15.0, 4.0 / 3.0, -2.0, 1.0]),
            ),
            # without axis, take indexes the flattened array
            dict(
                func_name="take",
                args=(gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), gs.array([[0, 5]])),
                expected=gs.array([[1.0, 6.0]]),
            ),
            dict(
                func_name="argmax",
                args=(gs.array([False, True, False]),),
                expected=gs.array(1),
            ),
            dict(
                func_name="clip",
                args=(gs.array([0.0, 1.0, 2.0]),),
                kwargs={"min": 0.5, "max": 1.5},
                expected=gs.array([0.5, 1.0, 1.5]),
            ),
            # repeated indices are summed
            dict(
                func_name="scatter_add",
                args=(gs.zeros(3),),
                kwargs={
                    "dim": 0,
                    "index": gs.array([0, 2, 0]),
                    "src": gs.array([1.0, 2.0, 3.0]),
                },
                expected=gs.array([4.0, 0.0, 2.0]),
            ),
            dict(
                func_name="scatter_add",
                args=(gs.ones((2, 3)),),
                kwargs={
                    "dim": -1,
                    "index": gs.array([[0, 0, 1], [2, 2, 2]]),
                    "src": gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                },
                expected=gs.array([[4.0, 4.0, 1.0], [1.0, 1.0, 16.0]]),
            ),
            dict(
                func_name="expand_dims_to",
                args=(gs.array([1.0, 2.0]), 2),
                expected=gs.array([[1.0, 2.0]]),
            ),
            dict(
                func_name="expand_dims_to",
                args=(gs.array([1.0, 2.0]), 2),
                kwargs={"axis": -1},
                expected=gs.array([[1.0], [2.0]]),
            ),
            dict(
                func_name="diagonal",
                args=(gs.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),),
                expected=gs.array([[1.0, 4.0], [5.0, 8.0]]),
            ),
            dict(
                func_name="diagonal",
                args=(gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),),
                kwargs={"offset": -1},
                expected=gs.array([4.0]),
            ),
            dict(
                func_name="trace",
                args=(gs.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),),
                expected=gs.array([5.0, 13.0]),
            ),
            dict(
                func_name="diag_embed",
                args=(gs.array([1.0, 2.0]),),
                kwargs={"offset": 1},
                expected=gs.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]),
            ),
            dict(
                func_name="diag_embed",
                args=(gs.array([1.0, 2.0]),),
                kwargs={"offset": -1},
                expected=gs.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            ),
            dict(
                func_name="linalg.vector_norm",
                args=(gs.array([[3.0, 4.0], [6.0, 8.0]]),),
                kwargs={"axis": -1},
                expected=gs.array([5.0, 10.0]),
            ),
            # by default, the vector norm is taken over all axes
            dict(
                func_name="linalg.vector_norm",
                args=(gs.array([[3.0, 0.0], [0.0, 4.0]]),),
                expected=gs.array(5.0),
            ),
            dict(
                func_name="linalg.matrix_norm",
                args=(gs.array([[[3.0, 0.0], [0.0, 4.0]]]),),
                expected=gs.array([5.0]),
            ),
            # vecdot conjugates its first argument
            dict(
                func_name="vecdot",
                args=(gs.array([1j, 1.0]), gs.array([1j, 1.0])),
                expected=gs.array(2.0 + 0j),
            ),
            dict(
                func_name="matrix_transpose",
                args=(gs.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),),
                expected=gs.array([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]),
            ),
            dict(
                func_name="array_from_sparse",
                kwargs={"indices": [(0,)], "data": [1.0], "target_shape": (2,)},
                expected=gs.array([1.0, 0.0]),
            ),
            dict(
                func_name="array_from_sparse",
                kwargs={
                    "indices": [(0, 0), (0, 1), (1, 2)],
                    "data": [1.0, 2.0, 3.0],
                    "target_shape": (2, 3),
                },
                expected=gs.array([[1.0, 2.0, 0], [0, 0, 3.0]]),
            ),
            dict(
                func_name="array_from_sparse",
                kwargs={
                    "indices": [(0, 1, 1), (1, 1, 1)],
                    "data": [1.0, 2.0],
                    "target_shape": (2, 2, 2),
                },
                expected=gs.array([[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 2.0]]]),
            ),
            # the triangular part is filled row by row
            dict(
                func_name="tril_embed",
                args=(gs.array([1.0, 2.0, 3.0]),),
                expected=gs.array([[1.0, 0.0], [2.0, 3.0]]),
            ),
            dict(
                func_name="tril_embed",
                args=(gs.array([1.0, 2.0, 3.0]),),
                kwargs={"k": -1},
                expected=gs.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 3.0, 0.0]]),
            ),
            dict(
                func_name="triu_embed",
                args=(gs.array([1.0, 2.0, 3.0]),),
                expected=gs.array([[1.0, 2.0], [0.0, 3.0]]),
            ),
            dict(
                func_name="triu_embed",
                args=(gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),),
                kwargs={"k": 1},
                expected=gs.array(
                    [
                        [[0.0, 1.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
                        [[0.0, 4.0, 5.0], [0.0, 0.0, 6.0], [0.0, 0.0, 0.0]],
                    ]
                ),
            ),
            dict(
                func_name="linalg.logm",
                args=[gs.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])],
                expected=gs.array(
                    [
                        [0.693147180, 0.0, 0.0],
                        [0.0, 1.098612288, 0.0],
                        [0.0, 0.0, 1.38629436],
                    ]
                ),
            ),
            dict(
                func_name="linalg.logm",
                args=[gs.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])],
                expected=gs.array(
                    [[0.0, 0.0, 0.0], [0.0, 1.609437912, 0.0], [0.0, 0.0, 1.79175946]]
                ),
            ),
            dict(
                func_name="linalg.expm",
                args=[
                    gs.array(
                        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
                    )
                ],
                expected=gs.array(
                    [
                        [7.38905609, 0.0, 0.0],
                        [0.0, 20.0855369, 0.0],
                        [0.0, 0.0, 54.5981500],
                    ]
                ),
            ),
            # TODO: uncomment or delete?
            # dict(
            #     func_name="linalg.expm",
            #     args=[gs.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])],
            #     expected=gs.array(
            #         [
            #             [2.718281828, 0.0, 0.0],
            #             [0.0, 148.413159, 0.0],
            #             [0.0, 0.0, 403.42879349],
            #         ],
            #     ),
            # ),
        ]
        smoke_data += self._take_data()

        return self.generate_tests(smoke_data)

    def func_out_equal_test_data(self):
        smoke_data = [
            dict(func_name="shape", args=(1,), expected=()),
            dict(func_name="shape", args=([1, 2],), expected=(2,)),
            dict(func_name="shape", args=(gs.ones(3),), expected=(3,)),
            dict(func_name="shape", args=(gs.ones((3, 3)),), expected=(3, 3)),
            dict(func_name="take", args=[gs.array([0, 1]), 0], expected=0),
        ]

        return self.generate_tests(smoke_data)
