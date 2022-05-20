import random

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.stratified.graph_space import (
    Graph,
    GraphSpace,
    GraphSpaceMetric,
    _vectorize_graph,
    _vectorize_graph_to_points,
)
from tests.data_generation import (
    TestData,
    _PointMetricTestData,
    _PointSetTestData,
    _PointTestData,
)


class GraphSpaceTestData(_PointSetTestData):

    _PointSet = GraphSpace
    _Point = Graph

    n_samples = 2
    n_points_list = random.sample(range(1, 5), n_samples)
    nodes_list = random.sample(range(1, 3), n_samples)
    space_args_list = [(n_nodes,) for n_nodes in nodes_list]

    def belongs_test_data(self):
        smoke_data = [
            dict(space_args=(3,), points=gs.ones((4, 4)), expected=False),
            dict(space_args=(3,), points=gs.ones((2, 2)), expected=True),
            dict(
                space_args=(2,),
                points=[self._Point(gs.ones((n + 1, n + 1))) for n in range(3)],
                expected=gs.array([True] * 2 + [False]),
            ),
        ]

        return self.generate_tests(smoke_data)

    def set_to_array_test_data(self):
        smoke_data = [
            dict(
                space_args=(2,),
                points=Graph(adj=gs.array([[1.0, 2.0], [3.0, 4.0]])),
                expected=gs.array([[1.0, 2.0], [3.0, 4.0]]),
            ),
        ]

        return self.generate_tests(smoke_data)

    def set_to_networkx_test_data(self):
        smoke_data = [
            dict(
                space=self._PointSet(2),
                points=Matrices(2, 2).random_point(),
            ),
        ]

        return self.generate_tests(smoke_data)

    def permute_test_data(self):
        space = self._PointSet(2)
        graph = gs.array([[0.0, 1.0], [2.0, 3]])

        smoke_data = [
            dict(
                space=space, graph=graph, permutation=gs.array([0, 1]), expected=graph
            ),
            dict(
                space=space,
                graph=graph,
                permutation=gs.array([1, 0]),
                expected=gs.array([[3.0, 2.0], [1.0, 0.0]]),
            ),
        ]

        return self.generate_tests(smoke_data)

    def permute_vectorization_test_data(self):
        space = self._PointSet(2)
        points = space.random_point(3)

        permutation = gs.array([0, 1])

        smoke_data = [
            dict(space=space, graph=points[0], id_permutation=permutation),
            dict(
                space=space,
                graph=points[0],
                id_permutation=gs.repeat(gs.expand_dims(permutation, 0), 2, axis=0),
            ),
            dict(space=space, graph=points, id_permutation=permutation),
            dict(
                space=space,
                graph=points,
                id_permutation=gs.repeat(
                    gs.expand_dims(permutation, 0), points.shape[0], axis=0
                ),
            ),
        ]

        return self.generate_tests(smoke_data)

    def pad_with_zeros_test_data(self):

        space = self._PointSet(3)

        adj_2 = Matrices(2, 2).random_point(3)
        adj_3 = Matrices(3, 3).random_point(2)
        points = [self._Point(adj_2[0]), self._Point(adj_3[0])]

        smoke_data = [
            dict(space=space, points=adj_2),
            dict(space=space, points=adj_2[0]),
            dict(space=space, points=adj_3),
            dict(space=space, points=adj_3[0]),
            dict(space=space, points=points),
            dict(space=space, points=points[0]),
        ]

        return self.generate_tests(smoke_data)


class GraphTestData(_PointTestData):
    _Point = Graph

    def to_array_test_data(self):
        smoke_data = [
            dict(
                point_args=(gs.array([[1.0, 2.0], [3.0, 4.0]]),),
                expected=gs.array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            dict(
                point_args=(gs.array([[1.0, 2.0], [3.0, 4.0]]),),
                expected=gs.array([[1.0, 2.0], [3.0, 4.0]]),
            ),
        ]

        return self.generate_tests(smoke_data)

    def to_networkx_test_data(self):
        adj = Matrices(3, 3).random_point()
        point = self._Point(adj)

        smoke_data = [dict(point=point)]

        return self.generate_tests(smoke_data)


class GraphSpaceMetricTestData(_PointMetricTestData):
    _PointSetMetric = GraphSpaceMetric
    _PointSet = GraphSpace
    _Point = Graph

    n_samples = 2
    n_nodes_list = random.sample(range(2, 4), n_samples)
    space_args_list = [(n_nodes,) for n_nodes in n_nodes_list]

    def dist_test_data(self):
        graph_a = Graph(adj=gs.array([[1.0, 2.0], [3.0, 4.0]]))
        graph_b = Graph(adj=gs.array([[1.0, 8.0], [3.0, 4.0]]))
        graph_c = Graph(adj=gs.array([[1.0, 6.0], [3.0, 4.0]]))

        smoke_data = [
            dict(
                space_args=(2,),
                start_point=[graph_a, graph_a, graph_b],
                end_point=[graph_b, graph_c, graph_c],
                expected=gs.array(
                    [6.0, 4.0, 2.0],
                ),
            ),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_test_data(self):
        graph_a = Graph(adj=gs.array([[1.0, 2.0], [3.0, 4.0]]))
        graph_b = Graph(adj=gs.array([[1.0, 8.0], [3.0, 4.0]]))

        # TODO: change for t != 0 and t != 1
        smoke_data = [
            dict(
                space_args=(2,),
                start_point=graph_a,
                end_point=graph_b,
                t=0.0,
                expected=gs.array([[[1.0, 2.0], [3.0, 4.0]]]),
            ),
        ]

        return self.generate_tests(smoke_data)

    def matching_test_data(self):
        space = self._PointSet(n_nodes=2)
        metric = self._PointSetMetric(space)

        graph_a = Graph(adj=gs.array([[1.0, 2.0], [3.0, 4.0]]))
        graph_b = Graph(adj=gs.array([[3.0, 4.0], [1.0, 2.0]]))

        smoke_data = [
            dict(
                metric=metric,
                point_a=graph_a,
                point_b=graph_b,
                matcher="ID",
                expected=gs.array([0, 1]),
            ),
            dict(
                metric=metric,
                point_a=graph_a,
                point_b=graph_b,
                matcher="FAQ",
                expected=gs.array([1, 0]),
            ),
        ]

        return self.generate_tests(smoke_data)

    def matching_output_shape_test_data(self):
        space = self._PointSet(*self.space_args_list[0])
        metric = self._PointSetMetric(space)
        pts = space.random_point(3)

        smoke_data = []
        for matcher in ["ID", "FAQ"]:
            smoke_data.extend(
                [
                    dict(
                        metric=metric, point_a=pts[0], point_b=pts[0], matcher=matcher
                    ),
                    dict(metric=metric, point_a=pts, point_b=pts[0], matcher=matcher),
                    dict(metric=metric, point_a=pts[0], point_b=pts, matcher=matcher),
                    dict(metric=metric, point_a=pts, point_b=pts, matcher=matcher),
                ]
            )

        return self.generate_tests(smoke_data)

    def matching_raises_error_test_data(self):
        space = self._PointSet(*self.space_args_list[0])
        metric = self._PointSetMetric(space)
        pt = space.random_point()

        smoke_data = [
            dict(metric=metric, point_a=pt, point_b=pt, invalid_matcher="invalid")
        ]

        return self.generate_tests(smoke_data)


class DecoratorsTestData(TestData):
    _Point = Graph
    _PointSet = GraphSpace

    def _get_data(self, fnc):
        adj = Matrices(3, 3).random_point(2)
        points = [self._Point(adj_) for adj_ in adj]

        smoke_data = [
            dict(fnc=fnc, points=points[0]),
            dict(fnc=fnc, points=points),
            dict(fnc=fnc, points=adj[0]),
            dict(fnc=fnc, points=adj),
        ]

        return self.generate_tests(smoke_data)

    def vectorize_graph_test_data(self):
        @_vectorize_graph((0, "points"))
        def vec_example(points):
            return points

        return self._get_data(vec_example)

    def vectorize_graph_to_points_test_data(self):
        @_vectorize_graph_to_points((0, "points"))
        def vec_example(points):
            return points

        return self._get_data(vec_example)
