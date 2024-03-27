from geomstats.geometry.stratified.trees import ForestTopology, Split
from geomstats.test.data import TestData


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


class BaseTopologyTestData(TestData):
    def partition_test_data(self):
        # TODO: add false example
        smoke_data = [
            dict(
                st_a=ForestTopology(partition=((1, 0), (2,)), split_sets=((), ())),
                st_b=ForestTopology(partition=((2,), (0, 1)), split_sets=((), ())),
                expected=True,
            ),
            dict(
                st_a=ForestTopology(
                    partition=((1,), (0,), (2,)), split_sets=((), (), ())
                ),
                st_b=ForestTopology(
                    partition=((0,), (1,), (2,)), split_sets=((), (), ())
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
                st_a=ForestTopology(partition=((0, 1),), split_sets=split_sets1),
                st_b=ForestTopology(partition=((0, 1),), split_sets=((),)),
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
                st_a=ForestTopology(partition=((0, 1, 2, 3),), split_sets=split_sets1),
                st_b=ForestTopology(partition=((1, 2), (0, 3)), split_sets=split_sets2),
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
                st_a=ForestTopology(partition=((0, 1, 2, 3),), split_sets=split_sets1),
                st_b=ForestTopology(partition=((1, 2), (0, 3)), split_sets=split_sets2),
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
                st_a=ForestTopology(partition=((0, 1, 2, 3),), split_sets=split_sets1),
                st_b=ForestTopology(partition=((0, 1, 2, 3),), split_sets=split_sets2),
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
                st_a=ForestTopology(partition=((0, 1, 2, 3),), split_sets=split_sets1),
                st_b=ForestTopology(partition=((1, 2), (0, 3)), split_sets=split_sets2),
                expected=[False, False, False, False, False, True],
            )
        )

        return self.generate_tests(smoke_data)
