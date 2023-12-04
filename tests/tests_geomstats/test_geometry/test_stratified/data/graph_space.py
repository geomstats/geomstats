import geomstats.backend as gs
from geomstats.geometry.stratified.graph_space import GraphSpace
from geomstats.test.data import TestData
from geomstats.test.test_case import np_backend

from ...data.matrices import MatricesTestData
from .quotient import AlignerCmpTestData, QuotientMetricWithArrayTestData

IS_NOT_NP = not np_backend()


class GraphAlignerCmpTestData(AlignerCmpTestData):
    N_RANDOM_POINTS = [1, 2]
    trials = 5


class PointToGeodesicAlignerTestData(TestData):
    skip_all = IS_NOT_NP

    def align_vec_test_data(self):
        return self.generate_vec_data()

    def dist_vec_test_data(self):
        return self.generate_vec_data()

    def dist_along_geodesic_is_zero_test_data(self):
        return self.generate_random_data()


class GraphSpaceTestData(MatricesTestData):
    skip_all = IS_NOT_NP


class GraphSpaceQuotientMetricTestData(QuotientMetricWithArrayTestData):
    skip_all = IS_NOT_NP


class GraphSpaceQuotientMetric2TestData:
    # TODO: which aligner? if ID, delete
    def dist_test_data(self):
        graph_a = gs.array([[1.0, 2.0], [3.0, 4.0]])
        graph_b = gs.array([[1.0, 8.0], [3.0, 4.0]])
        graph_c = gs.array([[1.0, 6.0], [3.0, 4.0]])

        smoke_data = [
            dict(
                point_a=gs.stack([graph_a, graph_a, graph_b]),
                point_b=gs.stack([graph_b, graph_c, graph_c]),
                expected=gs.array(
                    [6.0, 4.0, 2.0],
                ),
            ),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_test_data(self):
        graph_a = gs.array([[1.0, 2.0], [3.0, 4.0]])
        graph_b = gs.array([[1.0, 8.0], [3.0, 4.0]])

        smoke_data = [
            dict(
                start_point=graph_a,
                end_point=graph_b,
                t=0.0,
                expected=gs.array([[[1.0, 2.0], [3.0, 4.0]]]),
            ),
        ]

        return self.generate_tests(smoke_data)

    def align_point_to_point_test_data(self):
        space = GraphSpace(n_nodes=2, equip=True)
        id_aligner = space.metric.aligner

        graph_a = gs.array([[1.0, 2.0], [3.0, 4.0]])
        graph_b = gs.array([[3.0, 4.0], [1.0, 2.0]])

        smoke_data = [
            dict(
                point_a=graph_a,
                point_b=graph_b,
                aligner=id_aligner,
                expected=graph_b.adj,
            ),
        ]

        return self.generate_tests(smoke_data)

    def align_point_to_geodesic_test_data(self):
        space = GraphSpace(2, equip=True)
        space.metric.set_point_to_geodesic_aligner("default", n_points=3)

        base_point, end_point = space.random_point(2)
        geodesic = space.metric.geodesic(base_point=base_point, end_point=end_point)

        s = gs.linspace(0.0, 1.0, num=3)
        points = geodesic(s)

        smoke_data = [
            dict(
                space=space,
                geodesic=geodesic,
                point=points,
                expected=points,
            )
        ]

        return self.generate_tests(smoke_data)
