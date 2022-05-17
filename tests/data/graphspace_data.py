import geomstats.backend as gs
from tests.data_generation import TestData


class GraphSpaceTestData(TestData):
    def belongs_test_data(self):
        smoke_data = [
            dict(
                n=2,
                mat=gs.array([[[3.0, -1.0], [-1.0, 3.0]], [[4.0, -6.0], [-1.0, 3.0]]]),
                expected=[True, True],
            ),
            dict(n=2, mat=gs.array([-1.0, -1.0]), expected=False),
        ]
        return self.generate_tests(smoke_data)

    def random_point_belongs_test_data(self):
        smoke_data = [dict(n=2, n_points=1), dict(n=2, n_points=10)]
        return self.generate_tests(smoke_data)

    def permute_test_data(self):
        smoke_data = [
            dict(
                n=2,
                graph=gs.array([[0.0, 1.0], [2.0, 3.0]]),
                permutation=[1, 0],
                expected=gs.array([[3.0, 2.0], [1.0, 0.0]]),
            )
        ]
        return self.generate_tests(smoke_data)


class GraphSpaceMetricTestData(TestData):
    def matchers_test_data(self):
        smoke_data = [
            dict(
                n=2,
                set1=gs.array([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 0.0], [0.0, 1.0]]]),
                set2=gs.array([[[3.0, 2.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]),
            )
        ]

        return self.generate_tests(smoke_data)
