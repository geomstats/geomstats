from polpo.testing.data import CaseData

import geomstats.backend as gs


class AgainstNumpyTestData(CaseData):
    def func_like_np_test_data(self):
        data = [
            dict(func_name="array", args=([],)),
            dict(func_name="array", args=(1.5,)),
            dict(func_name="array", args=(gs.array(1.5),)),
            dict(func_name="array", args=([gs.ones(2), gs.ones(2)],)),
            dict(func_name="array", args=([gs.ones(1), gs.ones(1)],)),
            dict(func_name="array", args=([gs.ones(2), [0, 0]],)),
            dict(
                func_name="linalg.matrix_rank",
                args=(gs.array([[1.0, -1.0], [1.0, -1.0], [0.0, 0.0]]),),
            ),
            dict(func_name="prod", args=(gs.random.rand(2, 3),)),
            dict(func_name="cumprod", args=(gs.random.rand(2, 3),)),
            dict(func_name="amax", args=(gs.random.rand(2, 3, 4), (0, 2))),
            dict(
                func_name="logical_and",
                args=(gs.array([True, True, False]), gs.array([True, False, False])),
            ),
            dict(func_name="logical_and", args=(gs.array([True, False]), True)),
            dict(
                func_name="where",
                args=(gs.array([True, False]), gs.ones(2), gs.array([1j, 2j])),
            ),
            dict(func_name="where", args=(gs.array([True, False]), gs.arange(2), 0.5)),
            dict(func_name="any", args=(gs.random.rand(2, 3, 4) > 0.5,)),
            dict(func_name="any", args=(gs.random.rand(2, 3, 4) > 0.5, (0, -1))),
            dict(func_name="any", args=(gs.array([[0.0, 1.0], [0.0, 0.0]]), 1)),
            dict(func_name="any", args=([False, True],)),
            dict(func_name="all", args=(gs.random.rand(2, 3, 4) > 0.5,)),
            dict(func_name="all", args=(gs.random.rand(2, 3, 4) > 0.5, (0, -1))),
            dict(func_name="all", args=(gs.array([[0.0, 1.0], [1.0, 1.0]]), 1)),
            dict(func_name="all", args=([False, True],)),
        ]
        return data
