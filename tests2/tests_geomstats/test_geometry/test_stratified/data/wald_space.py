import random

import geomstats.backend as gs
from geomstats.geometry.stratified.wald_space import Split, Topology, Wald, WaldSpace
from geomstats.test.data import TestData

from .point_set import PointSetTestData, PointTestData


class WaldTestData(PointTestData):
    _Point = Wald

    n_samples = 2
    n_labels_list = random.sample(range(2, 4), n_samples)
    point_args_list = [(n_labels, 0.9, 0.9) for n_labels in n_labels_list]

    def generate_wald_belongs_test_data(self):
        random_data = [
            dict(point_args=point_args) for point_args in self.point_args_list
        ]

        return self.generate_tests([], random_data)

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


class WaldSpaceTestData(PointSetTestData):
    _Point = Wald
    _PointSet = WaldSpace

    n_samples = 2
    n_points_list = random.sample(range(1, 5), n_samples)
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


class SplitTestData(TestData):
    def restrict_to_test_data(self):
        smoke_data = [
            dict(
                split=Split(part1=[2, 3], part2=[0, 1, 4]),
                subset={0, 1, 2},
                expected=Split(part1=[0, 1], part2=[2]),
            ),
            dict(
                split=Split(part1=[0, 1, 2, 3, 4, 5], part2=[6]),
                subset={0},
                expected=Split(part1=[], part2=[0]),
            ),
        ]

        return self.generate_tests(smoke_data)

    def part_contains_test_data(self):
        smoke_data = [
            dict(
                split=Split(part1=[0, 4], part2=[1, 2, 3]),
                subset={0, 2},
                expected=False,
            ),
            dict(
                split=Split(part1=[0, 1, 2, 3, 6, 7, 8, 9], part2=[4, 5]),
                subset={0, 1, 2},
                expected=True,
            ),
        ]

        return self.generate_tests(smoke_data)

    def separates_test_data(self):
        smoke_data = [
            dict(split=Split(part1=[0, 1], part2=[2]), u=[0, 1], v=[2], expected=True),
            dict(
                split=Split(part1=[0, 1, 2], part2=[3, 4]),
                u=[0, 1, 2],
                v=[2, 3, 4],
                expected=False,
            ),
            dict(split=Split(part1=[0, 1], part2=[2, 3]), u=1, v=3, expected=True),
            dict(
                split=Split(part1=[], part2=[0, 1, 2, 3, 4]), u=4, v=1, expected=False
            ),
        ]

        return self.generate_tests(smoke_data)

    def get_part_towards_test_data(self):
        split1 = Split(part1=[0, 4], part2=[1, 2, 3])
        split2 = Split(part1=[2, 3], part2=[0, 1, 4])

        smoke_data = [
            dict(split_a=split1, split_b=split2, expected={1, 2, 3}),
            dict(split_a=split2, split_b=split1, expected={0, 1, 4}),
        ]

        return self.generate_tests(smoke_data)

    def get_part_away_from_test_data(self):
        split1 = Split(part1=[0, 4], part2=[1, 2, 3])
        split2 = Split(part1=[2, 3], part2=[0, 1, 4])

        smoke_data = [
            dict(
                split_a=split1,
                split_b=split2,
                expected={0, 4},
            ),
            dict(
                split_a=split2,
                split_b=split1,
                expected={2, 3},
            ),
        ]

        return self.generate_tests(smoke_data)

    def is_compatible_test_data(self):
        # TODO: add a non compatible example

        smoke_data = [
            dict(
                split_a=Split(part1=[0, 4], part2=[1, 2, 3]),
                split_b=Split(part1=[2, 3], part2=[0, 1, 4]),
                expected=True,
            ),
        ]

        return self.generate_tests(smoke_data)

    def hash_test_data(self):
        smoke_data = [
            dict(
                split_a=Split(part1=[0, 4], part2=[1, 2, 3]),
                split_b=Split(part1=[2, 3], part2=[0, 1, 4]),
                expected=False,
            ),
            dict(
                split_a=Split(part1=[0], part2=[1, 2, 3]),
                split_b=Split(part1=[0], part2=[1, 3, 2]),
                expected=True,
            ),
            dict(
                split_a=Split(part1=[2, 1], part2=[0, 3, 4]),
                split_b=Split(part1=[0, 4, 3], part2=[1, 2]),
                expected=True,
            ),
        ]

        return self.generate_tests(smoke_data)


class TopologyTestData(TestData):
    def partition_test_data(self):
        # TODO: add false example
        smoke_data = [
            dict(
                st_a=Topology(
                    n_labels=3, partition=((1, 0), (2,)), split_sets=((), ())
                ),
                st_b=Topology(
                    n_labels=3, partition=((2,), (0, 1)), split_sets=((), ())
                ),
                expected=True,
            ),
            dict(
                st_a=Topology(
                    n_labels=3, partition=((1,), (0,), (2,)), split_sets=((), (), ())
                ),
                st_b=Topology(
                    n_labels=3, partition=((0,), (1,), (2,)), split_sets=((), (), ())
                ),
                expected=True,
            ),
        ]

        return self.generate_tests(smoke_data)

    def partial_ordering_test_data(self):
        smoke_data = []

        sp1 = [[((0,), (1,))]]
        split_sets1 = [[Split(part1=a, part2=b) for a, b in splits] for splits in sp1]

        smoke_data.append(
            dict(
                st_a=Topology(n_labels=2, partition=((0, 1),), split_sets=split_sets1),
                st_b=Topology(n_labels=2, partition=((0, 1),), split_sets=((),)),
                expected=[True, True, False, False, False, True],
            )
        )

        sp1 = [
            [
                ((0,), (1, 2, 3)),
                ((3,), (0, 1, 2)),
                ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)),
                ((1, 2), (0, 3)),
            ]
        ]
        split_sets1 = [[Split(part1=a, part2=b) for a, b in splits] for splits in sp1]
        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(a, b) for a, b in splits] for splits in sp2]

        smoke_data.append(
            dict(
                st_a=Topology(
                    n_labels=4, partition=((0, 1, 2, 3),), split_sets=split_sets1
                ),
                st_b=Topology(
                    n_labels=4, partition=((1, 2), (0, 3)), split_sets=split_sets2
                ),
                expected=[True, True, False, False, False, True],
            )
        )

        sp1 = [
            [
                ((0,), (1, 2, 3)),
                ((3,), (0, 1, 2)),
                ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)),
                ((0, 2), (1, 3)),
            ]
        ]
        split_sets1 = [[Split(part1=a, part2=b) for a, b in splits] for splits in sp1]
        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(part1=a, part2=b) for a, b in splits] for splits in sp2]

        smoke_data.append(
            dict(
                st_a=Topology(
                    n_labels=4, partition=((0, 1, 2, 3),), split_sets=split_sets1
                ),
                st_b=Topology(
                    n_labels=4, partition=((1, 2), (0, 3)), split_sets=split_sets2
                ),
                expected=[False, False, False, False, False, True],
            )
        )

        sp1 = [
            [
                ((0,), (1, 2, 3)),
                ((3,), (0, 1, 2)),
                ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)),
                ((0, 2), (1, 3)),
            ]
        ]
        split_sets1 = [[Split(part1=a, part2=b) for a, b in splits] for splits in sp1]
        sp2 = [
            [
                ((0,), (1, 2, 3)),
                ((3,), (0, 1, 2)),
                ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)),
                ((0, 3), (1, 2)),
            ]
        ]
        split_sets2 = [[Split(part1=a, part2=b) for a, b in splits] for splits in sp2]

        smoke_data.append(
            dict(
                st_a=Topology(
                    n_labels=4, partition=((0, 1, 2, 3),), split_sets=split_sets1
                ),
                st_b=Topology(
                    n_labels=4, partition=((0, 1, 2, 3),), split_sets=split_sets2
                ),
                expected=[False, False, False, False, False, True],
            )
        )

        sp1 = [
            [((0,), (1, 2, 3)), ((3,), (0, 1, 2)), ((1,), (0, 2, 3)), ((2,), (0, 1, 3))]
        ]
        split_sets1 = [[Split(part1=a, part2=b) for a, b in splits] for splits in sp1]
        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(part1=a, part2=b) for a, b in splits] for splits in sp2]

        smoke_data.append(
            dict(
                st_a=Topology(
                    n_labels=4, partition=((0, 1, 2, 3),), split_sets=split_sets1
                ),
                st_b=Topology(
                    n_labels=4, partition=((1, 2), (0, 3)), split_sets=split_sets2
                ),
                expected=[False, False, False, False, False, True],
            )
        )

        return self.generate_tests(smoke_data)
