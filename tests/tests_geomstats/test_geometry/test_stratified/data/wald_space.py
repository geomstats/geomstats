import random

import geomstats.backend as gs
from geomstats.geometry.stratified.trees import Split
from geomstats.geometry.stratified.wald_space import Topology, Wald, WaldSpace
from tests.data_generation import _PointSetTestData, _PointTestData


class WaldTestData(_PointTestData):
    _Point = Wald

    def to_array_test_data(self):
        smoke_data = []

        top = Topology(n_labels=2, partition=((0,), (1,)), split_sets=((), ()))
        x = gs.array([])
        smoke_data.append(dict(point_args=(top, x), expected=gs.eye(2)))

        top = Topology(
            n_labels=3, partition=((0,), (1,), (2,)), split_sets=((), (), ())
        )
        x = gs.array([])
        smoke_data.append(dict(point_args=(top, x), expected=gs.eye(3)))

        top = Topology(
            n_labels=4, partition=((0,), (1,), (2,), (3,)), split_sets=((),) * 4
        )
        x = gs.array([])
        smoke_data.append(dict(point_args=(top, x), expected=gs.eye(4)))

        partition = ((0, 1, 2),)
        split_sets = ((((0, 1), (2,)), ((0, 2), (1,)), ((0,), (1, 2))),)
        split_sets = [[Split(a, b) for a, b in splits] for splits in split_sets]
        top = Topology(n_labels=3, partition=partition, split_sets=split_sets)
        x = gs.array([0.1, 0.2, 0.3])
        expected = gs.array([[1.0, 0.56, 0.63], [0.56, 1.0, 0.72], [0.63, 0.72, 1.0]])
        smoke_data.append(dict(point_args=(top, x), expected=expected))

        return self.generate_tests(smoke_data)


class WaldSpaceTestData(_PointSetTestData):

    _Point = Wald
    _PointSet = WaldSpace

    n_samples = 2
    n_points_list = [1] + random.sample(range(2, 5), n_samples - 1)
    n_labels_list = random.sample(range(2, 4), n_samples)
    space_args_list = [(n_labels,) for n_labels in n_labels_list]

    def belongs_test_data(self):
        smoke_data = []

        split_sets = ((((0,), (1,)),),)
        split_sets = [[Split(a, b) for a, b in splits] for splits in split_sets]
        top = Topology(n_labels=2, partition=((0, 1),), split_sets=split_sets)
        x = gs.array([0.3])
        point = self._Point(topology=top, weights=x)

        smoke_data.append(dict(space_args=(top.n_labels,), points=point, expected=True))
        smoke_data.append(dict(space_args=(3,), points=point, expected=False))

        top = Topology(n_labels=2, partition=((0, 1),), split_sets=((),))
        x = gs.array([])
        point = Wald(topology=top, weights=x)

        smoke_data.append(
            dict(space_args=(top.n_labels,), points=point, expected=False)
        )
        smoke_data.append(dict(space_args=(3,), points=point, expected=False))
        smoke_data.append(
            dict(space_args=(top.n_labels,), points=[point], expected=False)
        )
        smoke_data.append(
            dict(space_args=(top.n_labels,), points=point, expected=[False])
        )
        smoke_data.append(
            dict(space_args=(top.n_labels,), points=[point], expected=[False])
        )

        partition = ((0, 1, 2),)
        split_sets = ((((0, 1), (2,)), ((0, 2), (1,))),)
        split_sets = [[Split(a, b) for a, b in splits] for splits in split_sets]
        top = Topology(n_labels=3, partition=partition, split_sets=split_sets)
        x = gs.array([0.1, 0.2])
        point = Wald(topology=top, weights=x)
        smoke_data.append(dict(space_args=(top.n_labels,), points=point, expected=True))

        partition = ((0, 1, 2),)
        split_sets = ((((0, 1), (2,)),),)
        split_sets = [[Split(a, b) for a, b in splits] for splits in split_sets]
        top = Topology(n_labels=3, partition=partition, split_sets=split_sets)
        x = gs.array([0.1])
        point = Wald(topology=top, weights=x)
        smoke_data.append(
            dict(space_args=(top.n_labels,), points=point, expected=False)
        )

        partition = ((0, 1, 2),)
        split_sets = ((((0, 1), (2,)), ((0, 2), (1,))),)
        split_sets = [[Split(a, b) for a, b in splits] for splits in split_sets]
        top = Topology(n_labels=3, partition=partition, split_sets=split_sets)
        x = gs.array([0.0, 0.2])
        point = Wald(topology=top, weights=x)
        smoke_data.append(
            dict(space_args=(top.n_labels,), points=point, expected=False)
        )

        partition = ((0, 1, 2),)
        split_sets = ((((0, 1), (2,)), ((0, 2), (1,))),)
        split_sets = [[Split(a, b) for a, b in splits] for splits in split_sets]
        top = Topology(n_labels=3, partition=partition, split_sets=split_sets)
        x = gs.array([0.1, 1.0])
        point = Wald(topology=top, weights=x)
        smoke_data.append(
            dict(space_args=(top.n_labels,), points=point, expected=False)
        )

        points = []
        expected = []

        partition = ((0, 1, 2),)
        split_sets = ((((0, 1), (2,)), ((0, 2), (1,))),)
        split_sets = [[Split(a, b) for a, b in splits] for splits in split_sets]
        top = Topology(n_labels=3, partition=partition, split_sets=split_sets)
        x = gs.array([0.1, 0.2])
        point = Wald(topology=top, weights=x)
        points.append(point)
        expected.append(True)

        partition = ((0, 1, 2),)
        split_sets = ((((0, 1), (2,)),),)
        split_sets = [[Split(a, b) for a, b in splits] for splits in split_sets]
        top = Topology(n_labels=3, partition=partition, split_sets=split_sets)
        x = gs.array([0.1])
        point = Wald(topology=top, weights=x)
        points.append(point)
        expected.append(False)

        smoke_data.append(dict(space_args=(3,), points=points, expected=expected))
        return self.generate_tests(smoke_data)

    def set_to_array_test_data(self):
        smoke_data = []

        points = []
        expected = []

        top = Topology(
            n_labels=3, partition=((0,), (1,), (2,)), split_sets=((), (), ())
        )
        x = gs.array([])
        point = Wald(top, x)
        points.append(point)
        expected.append(gs.eye(3))

        partition = ((0, 1, 2),)
        split_sets = ((((0, 1), (2,)), ((0, 2), (1,)), ((0,), (1, 2))),)
        split_sets = [[Split(a, b) for a, b in splits] for splits in split_sets]
        top = Topology(n_labels=3, partition=partition, split_sets=split_sets)
        x = gs.array([0.1, 0.2, 0.3])
        point = Wald(top, x)
        points.append(point)
        expected.append(
            gs.array([[1.0, 0.56, 0.63], [0.56, 1.0, 0.72], [0.63, 0.72, 1.0]])
        )

        smoke_data.append(dict(space_args=(3,), points=points, expected=expected))

        return self.generate_tests(smoke_data)
