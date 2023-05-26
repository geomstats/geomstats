import numpy as np

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.data_generation import TestData

rand = gs.random.rand


class BackendsTestData(TestData):
    def _array_data(self):
        func_name = "array"

        args = [
            ([],),
            (1.5,),
            (gs.array(1.5),),
            ([gs.ones(2), gs.ones(2)],),
            ([gs.ones(1), gs.ones(1)],),
            ([gs.ones(2), [0, 0]],),
        ]

        return [{"func_name": func_name, "args": args_} for args_ in args]

    def _additional_array_data(self):
        data = [
            dict(func_name="zeros", args=(2,)),
            dict(func_name="zeros", args=((2, 2),)),
            dict(func_name="ones", args=(2,)),
            dict(func_name="ones", args=((2, 2),)),
        ]

        return data

    def array_like_np_test_data(self):
        smoke_data = []

        smoke_data += self._array_data()
        smoke_data += self._additional_array_data()

        return self.generate_tests(smoke_data)

    def _einsum_data(self):
        func_name = "einsum"

        args = [
            ("...i,...i->...", rand(2, 2), rand(2, 2)),
            ("...i,...i->...", rand(2, 2), rand(2)),
            ("...i,...i->...", rand(2), rand(2, 2)),
            ("...i,...i->...", rand(2), rand(2)),
            ("...,...i->...i", rand(1), rand(1, 3)),
            ("...,...i->...i", rand(1), rand(3)),
            ("...,...i->...i", 5.0, rand(1, 3)),
            ("...,...i->...i", 5.0, rand(3)),
            ("...,...i->...i", rand(3), rand(1, 3)),
            ("...,...i->...i", rand(3), rand(3)),
            ("...ij,...ik->...jk", rand(3, 2, 2), rand(3, 2, 2)),
            ("...ij,...ik->...jk", rand(2, 2), rand(3, 2, 2)),
            ("...ij,...ik->...jk", rand(3, 2, 2), rand(2, 2)),
            ("...ij,...ik->...jk", rand(2, 2), rand(2, 2)),
            ("...i,...ijk->...jk", rand(3), rand(3, 3, 3)),
            ("...i,...ijk->...jk", rand(3), rand(1, 3, 3, 3)),
            ("...i,...ijk->...jk", rand(2, 3), rand(2, 3, 3, 3)),
            ("...i,...ijk->...jk", rand(2, 3), rand(3, 3, 3)),
            ("...k,...j,...i->...kji", rand(3), rand(3), rand(3)),
            ("...k,...j,...i->...kji", rand(2, 3), rand(3), rand(3)),
        ]
        return [{"func_name": func_name, "args": args_} for args_ in args]

    def _logm_expm_data(self, func_name="linalg.logm"):
        arrays = [
            Matrices.to_diagonal(rand(3, 3)),
            # TODO: uncomment or delete?
            # Matrices.to_symmetric(rand(3, 3)),
            # rand(3, 3),
        ]
        return [dict(func_name=func_name, a=array) for array in arrays]

    def func_like_np_test_data(self):
        smoke_data = []
        smoke_data += self._einsum_data()

        return self.generate_tests(smoke_data)

    def unary_op_like_np_test_data(self):
        smoke_data = [
            dict(func_name="trace", a=rand(2, 2)),
            dict(func_name="trace", a=rand(3, 3)),
            dict(func_name="linalg.cholesky", a=SPDMatrices(3).random_point()),
            dict(func_name="linalg.eigvalsh", a=SymmetricMatrices(3).random_point()),
            dict(
                func_name="linalg.matrix_rank",
                a=gs.array([[1.0, -1.0], [1.0, -1.0], [0.0, 0.0]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def unary_op_like_scipy_test_data(self):
        smoke_data = []
        smoke_data += self._logm_expm_data()
        smoke_data += self._logm_expm_data("linalg.expm")

        return self.generate_tests(smoke_data)

    def unary_op_vec_test_data(self):
        smoke_data = [
            dict(func_name="trace", a=rand(3, 3)),
            dict(func_name="linalg.cholesky", a=SPDMatrices(3).random_point()),
            dict(func_name="linalg.eigvalsh", a=SymmetricMatrices(3).random_point()),
        ]
        smoke_data += self._logm_expm_data()
        smoke_data += self._logm_expm_data("linalg.expm")

        return self.generate_tests(smoke_data)

    def binary_op_like_np_test_data(self):
        smoke_data = [
            dict(func_name="matmul", a=rand(2, 2), b=rand(2, 2)),
            dict(func_name="matmul", a=rand(2, 3), b=rand(3, 2)),
            dict(func_name="outer", a=rand(3), b=rand(3)),
            dict(func_name="outer", a=rand(3), b=rand(4)),
            dict(func_name="dot", a=rand(3), b=rand(3)),
            dict(func_name="cross", a=rand(3), b=rand(3)),
        ]

        return self.generate_tests(smoke_data)

    def binary_op_like_einsum_test_data(self):
        smoke_data = [
            dict(func_name="matvec", a=rand(3, 3), b=rand(3), einsum_expr="ij,j->i")
        ]

        return self.generate_tests(smoke_data)

    def binary_op_vec_test_data(self):
        smoke_data = [
            dict(func_name="matmul", a=rand(3, 4), b=rand(4, 3)),
            dict(func_name="matmul", a=rand(3, 3), b=rand(3, 3)),
            dict(func_name="outer", a=rand(3), b=rand(3)),
            dict(func_name="outer", a=rand(3), b=rand(4)),
            dict(func_name="matvec", a=rand(3, 3), b=rand(3)),
            dict(func_name="matvec", a=rand(4, 3), b=rand(3)),
            dict(func_name="dot", a=rand(3), b=rand(3)),
            dict(func_name="cross", a=rand(3), b=rand(3)),
        ]

        return self.generate_tests(smoke_data)

    def binary_op_vec_raises_error_test_data(self):
        return self.binary_op_vec_test_data()

    def binary_op_raises_error_test_data(self):
        smoke_data = [
            dict(func_name="matmul", a=rand(1), b=rand(1)),
            dict(func_name="matmul", a=rand(2, 3, 3), b=rand(2, 3)),
            dict(func_name="matmul", a=rand(2, 3, 3), b=rand(3, 3, 3)),
            dict(func_name="matvec", a=rand(3, 2), b=rand(3)),
            dict(func_name="dot", a=rand(4), b=rand(3)),
            dict(func_name="dot", a=rand(3, 4), b=rand(3)),
            dict(func_name="cross", a=rand(4), b=rand(4)),
        ]

        return self.generate_tests(smoke_data)

    def binary_op_runs_test_data(self):
        smoke_data = []

        return self.generate_tests(smoke_data)

    def _pad_data(self):
        func_name = "pad"

        n, m = 2, 3
        args = [
            (gs.ones((n, n)), [[0, 1], [0, 1]]),
            (gs.ones((n, n)), [[0, 1], [0, 0]]),
            (gs.ones((m, n, n)), [[0, 0], [0, 1], [0, 1]]),
        ]
        expected = [(n + 1, n + 1), (n + 1, n), (m, n + 1, n + 1)]

        return [
            {"func_name": func_name, "args": args_, "expected": expected_}
            for args_, expected_ in zip(args, expected)
        ]

    def func_out_shape_test_data(self):
        smoke_data = []

        smoke_data += self._pad_data()

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
            #      expected=gs.transpose(gs.array([[0, 2]] * 2 + [[1, 3]])),
            #      axis=1),
            # dict(func_name=func_name, args=[mat, 0], expected=gs.array([0, 2]),
            #      axis=1)
        ]

        return data

    def func_out_allclose_test_data(self):
        # TODO: verify dtype
        smoke_data = [
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
            dict(
                func_name="mat_from_diag_triu_tril",
                args=(gs.ones(2), gs.array([2.0]), gs.array([3.0])),
                expected=gs.array([[1.0, 2.0], [3.0, 1.0]]),
            ),
            dict(
                func_name="mat_from_diag_triu_tril",
                args=(gs.ones(3), gs.array([2.0, 3.0, 4.0]), gs.array([5.0, 6.0, 7.0])),
                expected=gs.array([[1.0, 2.0, 3.0], [5.0, 1.0, 4.0], [6.0, 7.0, 1.0]]),
            ),
            # TODO: add (proper) vectorization test
            dict(
                func_name="mat_from_diag_triu_tril",
                args=(
                    gs.ones((2, 2)),
                    gs.array([[2.0], [2.0]]),
                    gs.array([[3.0], [3.0]]),
                ),
                expected=gs.repeat(
                    gs.expand_dims(gs.array([[1.0, 2.0], [3.0, 1.0]]), 0), 2, axis=0
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

    def compose_with_inverse_test_data(self):
        smoke_data = [
            dict(
                func_name_1="linalg.logm",
                func_name_2="linalg.expm",
                a=Matrices.to_diagonal(rand(3, 3)),
            ),
            dict(
                func_name_1="linalg.expm",
                func_name_2="linalg.logm",
                a=Matrices.to_diagonal(rand(3, 3)),
            ),
            dict(
                func_name_1="linalg.logm",
                func_name_2="linalg.expm",
                a=SpecialOrthogonal(n=3).random_point(2),
            ),
        ]
        return self.generate_tests(smoke_data)


class DtypesTestData(TestData):
    def array_test_data(self):
        smoke_data = [
            dict(ls=[1.0, 2.0], global_dtype_str="float32", expected_dtype=gs.float32),
            dict(ls=[1.0, 2.0], global_dtype_str="float64", expected_dtype=gs.float64),
            # TODO: uncomment later
            # dict(ls=[1, 2], expected_dtype=gs.int64),
            dict(
                ls=[1.0 + 3.0j, 2.0j],
                global_dtype_str="float32",
                expected_dtype=gs.complex64,
            ),
            dict(
                ls=[1.0 + 3.0j, 2.0j],
                global_dtype_str="float64",
                expected_dtype=gs.complex128,
            ),
        ]
        return self.generate_tests(smoke_data)

    def array_creation_test_data(self):
        smoke_data = [
            dict(
                func_name="array_from_sparse", args=([(0, 0)], [1.0], (2, 2)), kwargs={}
            ),
        ]
        return self.generate_tests(smoke_data)

    def array_creation_with_dtype_test_data(self):
        smoke_data = [
            dict(func_name="array", args=([1.0, 2.0],)),
            # TODO: add int test to arange
            dict(func_name="arange", args=((0.0, 2.0))),
            dict(func_name="linspace", args=(0.0, 1.0)),
            dict(func_name="random.normal", kwargs={"size": 2}),
            dict(
                func_name="random.multivariate_normal",
                kwargs={
                    "mean": gs.ones(2),
                    "cov": SPDMatrices(2).random_point(),
                    "size": 2,
                },
            ),
            dict(func_name="random.uniform", kwargs={"size": 2}),
        ]

        return self.generate_tests(smoke_data)

    def array_creation_with_dtype_from_shape_test_data(self):
        smoke_data = [
            dict(func_name="eye", shape=2),
            dict(func_name="ones", shape=2),
            dict(func_name="zeros", shape=2),
            dict(func_name="empty", shape=2),
            dict(func_name="random.rand", shape=3),
        ]

        return self.generate_tests(smoke_data)

    def array_creation_with_dtype_from_array_test_data(self):
        smoke_data = [
            dict(func_name="empty_like", array_shape=2),
            dict(func_name="ones_like", array_shape=2),
            dict(func_name="zeros_like", array_shape=2),
        ]

        return self.generate_tests(smoke_data)

    def unary_op_float_input_test_data(self):
        smoke_data = [
            dict(func_name="abs", x=1.0),
            dict(func_name="angle", x=1.0),
            dict(func_name="arccos", x=0.1),
            dict(func_name="arccosh", x=0.1),
            dict(func_name="arcsin", x=0.1),
            dict(func_name="arctanh", x=0.1),
            dict(func_name="ceil", x=1.0),
            dict(func_name="cos", x=1.0),
            dict(func_name="cosh", x=1.0),
            dict(func_name="exp", x=1.0),
            dict(func_name="floor", x=1.0),
            dict(func_name="imag", x=1.0 + 0.0j),
            dict(func_name="log", x=1.0),
            dict(func_name="real", x=1.0),
            dict(func_name="sign", x=1.0),
            dict(func_name="sin", x=1.0),
            dict(func_name="sinh", x=1.0),
            dict(func_name="sqrt", x=1.0),
            dict(func_name="tan", x=0.5),
            dict(func_name="tanh", x=0.5),
        ]
        return self.generate_tests(smoke_data)

    def unary_op_with_dtype_given_shape_test_data(self):
        axis_kwargs = {"axis": 0}
        smoke_data = [
            dict(func_name="cumprod", array_shape=2),
            dict(func_name="cumsum", array_shape=2),
            dict(func_name="mean", array_shape=2),
            dict(func_name="mean", array_shape=(2, 2), kwargs=axis_kwargs),
            dict(func_name="sum", array_shape=2),
            dict(func_name="sum", array_shape=(2, 2), kwargs=axis_kwargs),
            dict(func_name="std", array_shape=2),
            dict(func_name="std", array_shape=(2, 2), kwargs=axis_kwargs),
        ]

        return self.generate_tests(smoke_data)

    def from_numpy_given_shape_test_data(self):
        smoke_data = [
            dict(array_shape=(2, 2), np_func_array=np.ones),
        ]
        return self.generate_tests(smoke_data)

    def to_numpy_given_shape_test_data(self):
        smoke_data = [
            dict(array_shape=(2, 2)),
        ]

        return self.generate_tests(smoke_data)

    def unary_op_given_shape_test_data(self):
        smoke_data = [
            dict(func_name="copy", array_shape=(2, 2)),
            dict(func_name="diagonal", array_shape=(2, 2)),
            dict(func_name="erf", array_shape=2),
            dict(func_name="flatten", array_shape=(2, 2)),
            # TODO: add test in BackendsTestData
            dict(func_name="get_slice", array_shape=(2, 2), kwargs={"indices": [(0,)]}),
            dict(func_name="trace", array_shape=(2, 2)),
            dict(func_name="tril", array_shape=(2, 2)),
            dict(func_name="tril", array_shape=(2, 2), kwargs={"k": -1}),
            dict(func_name="triu", array_shape=(2, 2)),
            dict(func_name="triu", array_shape=(2, 2), kwargs={"k": 1}),
            dict(func_name="tril_to_vec", array_shape=(2, 2)),
            dict(func_name="triu_to_vec", array_shape=(2, 2)),
            dict(func_name="vec_to_diag", array_shape=3),
            # TODO: add test in BackendsTestData
            dict(func_name="linalg.norm", array_shape=3),
        ]

        return self.generate_tests(smoke_data)

    def unary_op_mult_out_given_shape_test_data(self):
        smoke_data = [dict(func_name="linalg.qr", array_shape=(3, 3))]

        return self.generate_tests(smoke_data)

    def unary_op_given_array_test_data(self):
        def _create_spd():
            return SPDMatrices(2).random_point()

        def _create_diag():
            return Matrices.to_diagonal(rand(2, 2))

        def _create_sym():
            return SymmetricMatrices(2).random_point()

        smoke_data = [
            dict(func_name="linalg.cholesky", create_array=_create_spd),
            dict(func_name="linalg.eigvalsh", create_array=_create_sym),
            dict(func_name="linalg.expm", create_array=_create_diag),
            dict(func_name="linalg.expm", create_array=_create_spd),
            dict(func_name="linalg.logm", create_array=_create_diag),
            dict(func_name="linalg.logm", create_array=_create_spd),
            dict(func_name="linalg.sqrtm", create_array=_create_spd),
        ]

        return self.generate_tests(smoke_data)

    def binary_op_float_input_test_data(self):
        smoke_data = [
            dict(func_name="arctan2", x1=1.0, x2=0.5),
            dict(func_name="mod", x1=1.0, x2=2.0),
            dict(func_name="power", x1=1.0, x2=1.0),
        ]

        return self.generate_tests(smoke_data)

    def binary_op_given_shape_test_data(self):
        smoke_data = [
            dict(func_name="cross", shape_a=3, shape_b=3),
            dict(func_name="divide", shape_a=3, shape_b=3),
            dict(
                func_name="divide",
                shape_a=3,
                shape_b=3,
                func_b=gs.zeros,
                kwargs={"ignore_div_zero": True},
            ),
            dict(func_name="dot", shape_a=3, shape_b=3),
            dict(func_name="dot", shape_a=3, shape_b=(2, 3)),
            dict(func_name="dot", shape_a=(2, 3), shape_b=(2, 3)),
            dict(func_name="matmul", shape_a=(2, 2), shape_b=(2, 2)),
            dict(func_name="matmul", shape_a=(2, 3), shape_b=(3, 2)),
            dict(func_name="matvec", shape_a=(3, 3), shape_b=3),
            dict(func_name="matvec", shape_a=(3, 3), shape_b=(2, 3)),
            dict(func_name="matvec", shape_a=(2, 3, 3), shape_b=(2, 3)),
            dict(func_name="outer", shape_a=3, shape_b=3),
            dict(func_name="outer", shape_a=3, shape_b=(2, 3)),
            dict(func_name="outer", shape_a=(2, 3), shape_b=(2, 3)),
        ]

        return self.generate_tests(smoke_data)

    def ternary_op_given_shape_test_data(self):
        smoke_data = [
            # TODO: add test in BackendsTestData
            dict(
                func_name="mat_from_diag_triu_tril",
                shape_a=(2,),
                shape_b=(1,),
                shape_c=(1,),
            ),
        ]
        return self.generate_tests(smoke_data)

    def ternary_op_given_array_test_data(self):
        def _create_sylvester():
            a = SPDMatrices(3).random_point()
            b = a
            q = rand(3, 3)
            return a, b, q

        smoke_data = [
            dict(func_name="linalg.solve_sylvester", create_array=_create_sylvester)
        ]

        return self.generate_tests(smoke_data)

    def func_out_dtype_test_data(self):
        smoke_data = [
            # TODO: add additional test for int
            dict(
                func_name="where",
                args=([True, False], 20.0, 20.0),
                kwargs={},
                expected=gs.float64,
            ),
        ]

        return self.generate_tests(smoke_data)

    def solve_sylvester_test_data(self):
        smoke_data = [
            dict(
                shape_a=(3, 3),
                shape_b=(1, 1),
                shape_c=(3, 1),
            ),
        ]

        return self.generate_tests(smoke_data)

    def random_distrib_complex_test_data(self):
        smoke_data = [
            dict(
                func_name="random.rand",
                args=(2,),
            ),
            dict(func_name="random.uniform", kwargs={"size": (2,)}),
            dict(func_name="random.normal", kwargs={"size": (2,)}),
            dict(
                func_name="random.multivariate_normal",
                args=(gs.zeros(2), SPDMatrices(2).random_point()),
                kwargs={"size": (2,)},
            ),
        ]

        return self.generate_tests(smoke_data)
