import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.test.data import TestData

rand = gs.random.rand


class AgainstScipyTestData(TestData):
    def _logm_expm_data(self, func_name="linalg.logm"):
        arrays = [
            Matrices.to_diagonal(rand(3, 3)),
            # TODO: uncomment or delete?
            # Matrices.to_symmetric(rand(3, 3)),
            # rand(3, 3),
        ]
        return [dict(func_name=func_name, a=array) for array in arrays]

    def unary_op_like_scipy_test_data(self):
        smoke_data = []
        smoke_data += self._logm_expm_data()
        smoke_data += self._logm_expm_data("linalg.expm")

        return self.generate_tests(smoke_data)
