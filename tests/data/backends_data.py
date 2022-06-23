import geomstats.backend as gs
from tests.data_generation import TestData


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

    def np_like_array_test_data(self):

        smoke_data = []

        smoke_data += self._array_data()
        smoke_data += self._additional_array_data()

        return self.generate_tests(smoke_data)

    def np_like_binary_op_test_data(self):
        smoke_data = [
            dict(func_name="matmul", a=gs.random.rand(2, 2), b=gs.random.rand(2, 2)),
            dict(func_name="matmul", a=gs.random.rand(2, 3), b=gs.random.rand(3, 2)),
            dict(func_name="outer", a=gs.random.rand(3), b=gs.random.rand(3)),
            dict(func_name="outer", a=gs.random.rand(3), b=gs.random.rand(4)),
            dict(func_name="dot", a=gs.random.rand(3), b=gs.random.rand(3)),
        ]

        return self.generate_tests(smoke_data)

    def binary_op_vec_test_data(self):

        smoke_data = [
            dict(func_name="matmul", a=gs.random.rand(3, 4), b=gs.random.rand(4, 3)),
            dict(func_name="matmul", a=gs.random.rand(3, 3), b=gs.random.rand(3, 3)),
            dict(func_name="outer", a=gs.random.rand(3), b=gs.random.rand(3)),
            dict(func_name="outer", a=gs.random.rand(3), b=gs.random.rand(4)),
            dict(func_name="matvec", a=gs.random.rand(3, 3), b=gs.random.rand(3)),
            dict(func_name="matvec", a=gs.random.rand(4, 3), b=gs.random.rand(3)),
            dict(func_name="dot", a=gs.random.rand(3), b=gs.random.rand(3)),
        ]

        return self.generate_tests(smoke_data)

    def binary_raises_error_test_data(self):
        smoke_data = [
            dict(func_name="matmul", a=gs.random.rand(1), b=gs.random.rand(1)),
            dict(func_name="matmul", a=gs.random.rand(2), b=gs.random.rand(2)),
            dict(func_name="matvec", a=gs.random.rand(3, 2), b=gs.random.rand(3)),
            dict(func_name="matvec", a=gs.random.rand(2, 3, 3), b=gs.random.rand(4, 3)),
            dict(func_name="dot", a=gs.random.rand(4), b=gs.random.rand(3)),
            dict(func_name="dot", a=gs.random.rand(3, 4), b=gs.random.rand(3)),
            dict(func_name="dot", a=gs.random.rand(3, 4), b=gs.random.rand(2, 3)),
        ]

        return self.generate_tests(smoke_data)
