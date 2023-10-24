import geomstats.backend as gs
from geomstats.test.data import TestData

rand = gs.random.rand


class AgainstEinsumTestData(TestData):
    def binary_op_like_einsum_test_data(self):
        smoke_data = [
            dict(func_name="matvec", a=rand(3, 3), b=rand(3), einsum_expr="ij,j->i")
        ]
        return self.generate_tests(smoke_data)
