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

        def belongs_test_data(self):
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

            smoke_data = [
                dict(n=n, index=index, point=p1, expected=True),
                dict(n=n, index=index, point=p2, expected=True),
                dict(n=n, index=index, point=gs.array([p1, p2]), expected=2 * [True]),
                dict(n=n, index=index, point=p3, expected=False),  # same subspace
                dict(n=n, index=index, point=p4, expected=False),  # wrong trace
                dict(n=n, index=index, point=p5, expected=10 * [False])  # wrong trace
            ]
            return self.generate_tests(smoke_data)

        def is_tangent_test_data(self):
            n = 5
            index = [1, 3, 4]
            p1 = gs.array(
                [gs.array(np.diag([1, 0, 0, 0, 0])), gs.array(np.diag([0, 1, 1, 0, 0])),
                 gs.array(np.diag([0, 0, 0, 1, 0]))])
            v1 = gs.array(
                [gs.array(np.diag([1, 0, 0, 0, 0])), gs.array(np.diag([0, 1, 1, 0, 0])),
                 gs.array(np.diag([0, 0, 0, 1, 0]))])
            v2 = gs.zeros((len(index), n, n))

            smoke_data = [
                dict(n=n, index=index, vector=v1, base_point=p1, expected=False),
                dict(n=n, index=index, vector=v2, base_point=p1, expected=True),
                dict(n=n, index=index, vector=gs.array([v1, v2]), base_point=gs.array(
                    [p1, p1]), expected=[False, True])
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_test_data(self):
            pass

        def projection_belongs_test_data(self):
            pass

        def to_tangent_is_tangent_test_data(self):
            return []

        def random_tangent_vec_is_tangent_test_data(self):
            pass

    testing_data = TestDataFlag()

    def test_belongs(self, n, index, point, expected):
        self.assertAllClose(self.space(n, index).belongs(point), gs.array(expected))

    def test_is_tangent(self, n, index, vector, base_point, expected):
        self.assertAllClose(self.space(n, index).is_tangent(vector, base_point),
                            gs.array(expected))
