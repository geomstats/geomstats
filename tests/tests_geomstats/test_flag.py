import numpy as np

import geomstats.backend as gs
from geomstats.geometry.flag import Flag
from tests.conftest import Parametrizer
from tests.data_generation import _ManifoldTestData
from tests.geometry_test_cases import ManifoldTestCase


class TestFlag(ManifoldTestCase, metaclass=Parametrizer):
    space = Flag
    skip_test_random_point_belongs = True
    skip_test_projection_belongs = True
    skip_test_to_tangent_is_tangent_test_data = True
    skip_test_random_tangent_vec_is_tangent = True

    class TestDataFlag(_ManifoldTestData):
        n = 5
        index = [1, 3, 4]
        p1 = gs.array(
            [gs.array(np.diag([1, 0, 0, 0, 0])), gs.array(np.diag([0, 1, 1, 0, 0])),
             gs.array(np.diag([0, 0, 0, 1, 0]))])
        p2 = gs.array(
            [gs.array(np.diag([0, 1, 0, 0, 0])), gs.array(np.diag([1, 0, 0, 1, 0])),
             gs.array(np.diag([0, 0, 1, 0, 0]))])
        p3 = gs.array(
            [gs.array(np.diag([1, 0, 0, 0, 0])), gs.array(np.diag([1, 0, 1, 0, 0])),
             gs.array(np.diag([0, 0, 0, 1, 0]))])
        p4 = gs.array(
            [gs.array(np.diag([1, 0, 0, 0, 0])), gs.array(np.diag([0, 1, 1, 0, 0])),
             gs.array(np.diag([0, 0, 0, 1, 1]))])
        p5 = gs.zeros((10, len(index), n, n))

        def belongs_test_data(self):
            smoke_data = [
                dict(n=5, index=[1, 3, 4], point=p1, expected=True),
                dict(n=5, index=[1, 3, 4], point=p2, expected=True),
                dict(n=5, index=[1, 3, 4], point=gs.array([p1, p2]), expected=[
                    True, True]),
                dict(n=5, index=[1, 3, 4], point=p3, expected=False),
                dict(n=5, index=[1, 3, 4], point=p4, expected=False),
                dict(n=5, index=[1, 3, 4], point=p5, expected=10 * [False]),

            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_test_data(self):
            pass

        def projection_belongs_test_data(self):
            pass

        def to_tangent_is_tangent_test_data(self):
            pass

        def random_tangent_vec_is_tangent_test_data(self):
            pass

    testing_data = TestDataFlag()
