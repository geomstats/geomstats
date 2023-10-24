import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.test.data import TestData

rand = gs.random.rand


class AgainstNumpyTestData(TestData):
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
        data = []

        data += self._array_data()
        data += self._additional_array_data()

        return self.generate_tests(data)

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
        data = []
        data += self._einsum_data()

        return self.generate_tests(data)

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
