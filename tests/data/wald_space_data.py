import random

import geomstats.backend as gs
from geomstats.stratified_geometry.wald_space import Split, Topology
from tests.data_generation import _PointSetTestData, _PointTestData


class WaldTestData(_PointTestData):
    def to_array_test_data(self):
        smoke_data = []

        top = Topology(n=2, partition=((0,), (1,)), split_sets=((), ()))
        x = gs.array([])
        smoke_data.append(dict(point_args=(top.n, top, x), expected=gs.eye(2)))

        top = Topology(n=3, partition=((0,), (1,), (2,)), split_sets=((), (), ()))
        x = gs.array([])
        smoke_data.append(dict(point_args=(top.n, top, x), expected=gs.eye(3)))

        top = Topology(n=4, partition=((0,), (1,), (2,), (3,)), split_sets=((),) * 4)
        x = gs.array([])
        smoke_data.append(dict(point_args=(top.n, top, x), expected=gs.eye(4)))

        partition = ((0, 1, 2),)
        split_sets = ((((0, 1), (2,)), ((0, 2), (1,)), ((0,), (1, 2))),)
        split_sets = [[Split(a, b) for a, b in splits] for splits in split_sets]
        top = Topology(n=3, partition=partition, split_sets=split_sets)
        x = gs.array([0.1, 0.2, 0.3])
        expected = gs.array([[1.0, 0.56, 0.63], [0.56, 1.0, 0.72], [0.63, 0.72, 1.0]])
        smoke_data.append(dict(point_args=(top.n, top, x), expected=expected))

        return self.generate_tests(smoke_data)


class WaldSpaceTestData(_PointSetTestData):
    # for random tests
    n_samples = _PointSetTestData.n_samples
    n_labels_list = random.sample(range(2, 4), n_samples)
    space_args_list = [(n_labels,) for n_labels in n_labels_list]

    def belongs_test_data(self):
        smoke_data = []
        return self.generate_tests(smoke_data)

    def set_to_array_test_data(self):
        smoke_data = []
        return self.generate_tests(smoke_data)
