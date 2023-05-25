import random

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.stratified.graph_space import (
    ExhaustiveAligner,
    FAQAligner,
    GraphPoint,
    GraphSpace,
    GraphSpaceMetric,
    IDAligner,
    PointToGeodesicAligner,
    _GeodesicToPointAligner,
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
    _Point = GraphPoint

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
                points=GraphPoint(adj=gs.array([[1.0, 2.0], [3.0, 4.0]])),
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
        graph = gs.array([[0.0, 1.0], [2.0, 3.0]])

        smoke_data = [
            dict(
                space=space, graph=graph, permutation=gs.array([0, 1]), expected=graph
            ),
        ]
        vec_data = (
            dict(
                space=space,
                graph=graph,
                permutation=gs.array([1, 0]),
                expected=gs.array([[3.0, 2.0], [1.0, 0.0]]),
            ),
        )

        return self.generate_tests(smoke_data) + self.generate_vectorization_tests(
            vec_data,
            ["graph", "permutation"],
            expected_name="expected",
            n_reps=3,
        )

    def pad_with_zeros_test_data(self):
        space = self._PointSet(4)

        adjs = [
            Matrices(2, 2).random_point(),
            Matrices(3, 3).random_point(),
            Matrices(4, 4).random_point(),
        ]
        points = [self._Point(adj) for adj in adjs]

        vec_data_1 = [dict(space=space, points=point) for point in adjs]
        vec_data_2 = [dict(space=space, points=point) for point in points]

        return self.generate_vectorization_tests(
            vec_data_1, ["points"]
        ) + self.generate_vectorization_tests(
            vec_data_2, ["points"], check_expand=False
        )


class GraphTestData(_PointTestData):
    _Point = GraphPoint

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
    _Point = GraphPoint

    n_samples = 2
    n_nodes_list = random.sample(range(2, 4), n_samples)
    space_args_list = [(n_nodes,) for n_nodes in n_nodes_list]

    def dist_test_data(self):
        graph_a = GraphPoint(adj=gs.array([[1.0, 2.0], [3.0, 4.0]]))
        graph_b = GraphPoint(adj=gs.array([[1.0, 8.0], [3.0, 4.0]]))
        graph_c = GraphPoint(adj=gs.array([[1.0, 6.0], [3.0, 4.0]]))

        smoke_data = [
            dict(
                space_args=(2,),
                point_a=[graph_a, graph_a, graph_b],
                point_b=[graph_b, graph_c, graph_c],
                expected=gs.array(
                    [6.0, 4.0, 2.0],
                ),
            ),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_test_data(self):
        graph_a = GraphPoint(adj=gs.array([[1.0, 2.0], [3.0, 4.0]]))
        graph_b = GraphPoint(adj=gs.array([[1.0, 8.0], [3.0, 4.0]]))

        smoke_data = [
            dict(
                space=GraphSpace(n_nodes=2, equip=True),
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

        graph_a = GraphPoint(adj=gs.array([[1.0, 2.0], [3.0, 4.0]]))
        graph_b = GraphPoint(adj=gs.array([[3.0, 4.0], [1.0, 2.0]]))

        smoke_data = [
            dict(
                space=space,
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


class DecoratorsTestData(TestData):
    _Point = GraphPoint
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


class AlignerTestData(TestData):
    def __init__(self):
        self._setup()

    def _setup(self):
        self.space_args_list = [(3,)]

    def _get_aligners(self):
        Aligners = [IDAligner, FAQAligner, ExhaustiveAligner]
        return [Aligner() for Aligner in Aligners]

    def align_test_data(self):
        exhaustive_aligner = ExhaustiveAligner()

        smoke_data = []
        for space_args in self.space_args_list:
            space = GraphSpace(*space_args, equip=True)
            base_point, permute_point = space.random_point(2)

            expected = exhaustive_aligner.align(space, base_point, permute_point)

            aligners = [FAQAligner()]
            for aligner in aligners:
                space.metric.set_aligner(aligner)

                smoke_data.append(
                    dict(
                        space=space,
                        aligner=aligner,
                        base_point=base_point,
                        permute_point=permute_point,
                        expected=expected,
                        dist_fnc=space.metric.dist,
                    )
                )

        return self.generate_tests(smoke_data)

    def align_cmp_points_test_data(self):
        vec_data = []
        for space_args in self.space_args_list:
            space = GraphSpace(*space_args, equip=True)
            base_point, permute_point = space.random_point(2)

            aligners = self._get_aligners()
            for aligner in aligners:
                if isinstance(aligner, IDAligner):
                    expected = permute_point
                else:
                    expected = aligner.align(space, base_point, permute_point)

                vec_data.append(
                    dict(
                        space=space,
                        aligner=aligner,
                        base_point=base_point,
                        permute_point=permute_point,
                        expected=expected,
                    )
                )

        return self.generate_vectorization_tests(
            vec_data, ["base_point", "permute_point"], expected_name="expected"
        )

    def align_output_shape_test_data(self):
        smoke_data = []
        for space_args in self.space_args_list:
            space = GraphSpace(*space_args, equip=True)
            base_point, permute_point = space.random_point(2)

            aligners = self._get_aligners()
            for aligner in aligners:
                smoke_data.append(
                    dict(
                        space=space,
                        aligner=aligner,
                        base_point=base_point,
                        permute_point=permute_point,
                    )
                )

        return self.generate_vectorization_tests(
            smoke_data, ["base_point", "permute_point"]
        )


class PointToGeodesicAlignerTestData(TestData):
    tolerances = {
        "dist": {"atol": 1e-8},
    }

    def __init__(self):
        self._setup()

    def _setup(self):
        self.space_args_list = [(3,)]

    def _get_aligners_and_geo(self, space):
        space.metric.set_aligner(ExhaustiveAligner())
        init_point, end_point = space.random_point(2)

        geodesic = space.metric.geodesic(init_point, end_point)

        aligners = [
            PointToGeodesicAligner(s_min=0.0, s_max=1.0, n_points=3),
            _GeodesicToPointAligner(),
        ]
        return aligners, geodesic

    def align_test_data(self):
        smoke_data = []
        for space_args in self.space_args_list:
            space = GraphSpace(*space_args, equip=True)
            aligners, geodesic = self._get_aligners_and_geo(space)

            s = gs.linspace(0.0, 1.0, num=3)
            points = geodesic(s)

            for aligner in aligners:
                smoke_data.append(
                    dict(
                        space=space,
                        aligner=aligner,
                        geodesic=geodesic,
                        point=points,
                        expected=points,
                    )
                )

        vec_data = []
        for space_args in self.space_args_list:
            space = GraphSpace(*space_args, equip=True)
            aligners, geodesic = self._get_aligners_and_geo(space)
            point = geodesic(0.5)[0]

            for aligner in aligners:
                vec_data.append(
                    dict(
                        space=space,
                        aligner=aligner,
                        geodesic=geodesic,
                        point=point,
                        expected=point,
                    )
                )

        return self.generate_tests(smoke_data) + self.generate_vectorization_tests(
            vec_data, ["point"], expected_name="expected"
        )

    def dist_test_data(self):
        smoke_data = []
        for space_args in self.space_args_list:
            space = GraphSpace(*space_args, equip=True)
            aligners, geodesic = self._get_aligners_and_geo(space)

            s = gs.linspace(0.0, 1.0, num=3)
            points = geodesic(s)

            for aligner in aligners:
                smoke_data.append(
                    dict(
                        space=space,
                        aligner=aligner,
                        geodesic=geodesic,
                        point=points,
                        expected=0.0,
                    )
                )

        vec_data = []
        for space_args in self.space_args_list:
            space = GraphSpace(*space_args, equip=True)
            aligners, geodesic = self._get_aligners_and_geo(space)
            point = geodesic(0.5)[0]

            for aligner in aligners:
                vec_data.append(
                    dict(
                        space=space,
                        aligner=aligner,
                        geodesic=geodesic,
                        point=point,
                        expected=0.0,
                    )
                )

        return self.generate_tests(smoke_data) + self.generate_vectorization_tests(
            vec_data, ["point"], expected_name="expected"
        )
