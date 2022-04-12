import random

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.stratified_geometry.graphspace import Graph
from tests.data_generation import (
    _PointGeometryTestData,
    _PointSetTestData,
    _PointTestData,
)


class GraphSpaceTestData(_PointSetTestData):

    # for random tests
    n_samples = _PointSetTestData.n_samples
    nodes_list = random.sample(range(1, 3), n_samples)
    space_args_list = [(nodes,) for nodes in nodes_list]

    def belongs_test_data(self):
        g1 = Graph(adj=Matrices(3, 3).random_point(1))
        g2 = Matrices(4, 4).random_point(1)
        g3 = Matrices(4, 4).random_point(2)
        # g4 = [
        #   Graph(adj=Matrices(3, 3).random_point(1)),
        #    Graph(adj=Matrices(4, 4).random_point(1)),
        # ]

        smoke_data = [
            dict(space_args=(3,), points=g1, expected=[True]),
            dict(space_args=(4,), points=g2, expected=[True]),
            dict(space_args=(5,), points=g3, expected=[False, False]),
            # dict(space_args=(3,), points=g4, expected=[True, False]),
        ]

        return self.generate_tests(smoke_data)

    def set_to_array_test_data(self):
        g0 = Graph(adj=gs.array([[1.0, 2.0], [3.0, 4.0]]))
        g1 = [
            Graph(adj=gs.array([[1.0, 2.0], [3.0, 4.0]])),
            Graph(adj=gs.array([[1.0, 2.0], [3.0, 4.0]])),
        ]
        g2 = gs.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [1.0, 2.0, 5.0]])
        g3 = gs.array(
            [
                [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [1.0, 2.0, 5.0]],
                [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [1.0, 2.0, 5.0]],
            ]
        )
        smoke_data = [
            dict(
                space_args=(2,),
                points=g0,
                expected=gs.array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            dict(
                space_args=(2,),
                points=g1,
                expected=gs.array([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]),
            ),
            dict(
                space_args=(3,),
                points=g2,
                expected=gs.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [1.0, 2.0, 5.0]]),
            ),
            dict(
                space_args=(3,),
                points=g3,
                expected=gs.array(
                    [
                        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [1.0, 2.0, 5.0]],
                        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [1.0, 2.0, 5.0]],
                    ]
                ),
            ),
        ]

        return self.generate_tests(smoke_data)


class GraphTestData(_PointTestData):
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


class GraphSpaceGeometryTestData(_PointGeometryTestData):
    def dist_test_data(self):
        pts_start = [
            Graph(adj=gs.array([[1.0, 2.0], [3.0, 4.0]])),
            Graph(adj=gs.array([[1.0, 8.0], [3.0, 4.0]])),
        ]

        pts_end = [
            Graph(adj=gs.array([[1.0, 2.0], [3.0, 4.0]])),
            Graph(adj=gs.array([[1.0, 6.0], [3.0, 4.0]])),
        ]

        smoke_data = [
            dict(
                space_args=(2,),
                start_point=pts_start[0],
                end_point=pts_end[0],
                expected=gs.array([0.0]),
            ),
            dict(
                space_args=(2,),
                start_point=pts_start[0],
                end_point=pts_end,
                expected=gs.array([0.0, 4.0]),
            ),
            dict(
                space_args=(2,),
                start_point=pts_start,
                end_point=pts_end[0],
                expected=gs.array([0.0, 6.0]),
            ),
            dict(
                space_args=(2,),
                start_point=pts_start,
                end_point=pts_end,
                expected=gs.array([0.0, 2.0]),
            ),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_test_data(self):
        pts_start = [
            Graph(adj=gs.array([[1.0, 2.0], [3.0, 4.0]])),
            Graph(adj=gs.array([[1.0, 8.0], [3.0, 4.0]])),
        ]

        pts_end = [
            Graph(adj=gs.array([[1.0, 2.0], [3.0, 4.0]])),
            Graph(adj=gs.array([[1.0, 6.0], [3.0, 4.0]])),
        ]

        smoke_data = [
            dict(
                space_args=(2,),
                start_point=pts_start[0],
                end_point=pts_end[1],
                t=0.0,
                expected=gs.array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            dict(
                space_args=(12,),
                start_point=pts_start[0],
                end_point=pts_end[1],
                t=1,
                expected=gs.array([[1.0, 6.0], [3.0, 4.0]]),
            ),
            dict(
                space_args=(12,),
                start_point=pts_start[0],
                end_point=pts_end[1],
                t=[0.0, 1.0],
                expected=gs.array([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 6.0], [3.0, 4.0]]]),
            ),
            dict(
                space_args=(12,),
                start_point=pts_start,
                end_point=pts_end,
                t=[0.0],
                expected=gs.array([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]),
            ),
            dict(
                space_args=(12,),
                start_point=pts_start,
                end_point=pts_end,
                t=[0.0, 1.0],
                expected=gs.array([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 6.0], [3.0, 4.0]]]),
            ),
        ]

        return self.generate_tests(smoke_data)
