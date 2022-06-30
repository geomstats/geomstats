import geomstats.backend as gs
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

    def func_like_np_test_data(self):
        smoke_data = []
        smoke_data += self._einsum_data()

        return self.generate_tests(smoke_data)

    def unary_op_like_np_test_data(self):
        smoke_data = [
            dict(func_name="trace", a=rand(2, 2)),
            dict(func_name="trace", b=rand(3, 3)),
        ]
        return self.generate_tests(smoke_data)

    def unary_op_vec_test_data(self):
        smoke_data = [dict(func_name="trace", b=rand(3, 3))]
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

    def bool_unary_func_test_data(self):
        smoke_data = [
            dict(func_name="is_array", a=gs.ones(2), expected=True),
            dict(func_name="is_array", a=[1, 2], expected=False),
            dict(func_name="is_array", a=1, expected=False),
        ]

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

    def func_out_equal_test_data(self):
        smoke_data = [
            dict(func_name="shape", args=(1,), expected=()),
            dict(func_name="shape", args=([1, 2],), expected=(2,)),
            dict(func_name="shape", args=(gs.ones(3),), expected=(3,)),
            dict(func_name="shape", args=(gs.ones((3, 3)),), expected=(3, 3)),
        ]

        return self.generate_tests(smoke_data)
