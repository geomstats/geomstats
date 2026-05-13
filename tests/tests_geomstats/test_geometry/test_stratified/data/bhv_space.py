import geomstats.backend as gs
from geomstats.geometry.stratified.bhv_space import Split, Tree
from geomstats.test.data import TestData


class TreeTopologyTestData(TestData):
    def raises_empty_splits_test_data(self):
        data = [
            dict(
                invalid_splits=(Split({}, {0, 1, 2, 3, 4}), Split({0, 1}, {2, 3, 4})),
            ),
        ]
        return self.generate_tests(data)

    def raises_singleton_test_data(self):
        data = [
            dict(
                invalid_splits=(Split({3}, {0, 4, 1, 2}), Split({0, 1}, {2, 3, 4})),
            ),
        ]
        return self.generate_tests(data)

    def valid_topology_attributes_test_data(self):
        data = [
            dict(
                splits=(Split({3, 4}, {0, 1, 2}), Split({0, 1}, {2, 3, 4})),
                n_expected_labels=5,
            ),
            dict(
                splits=(),
                n_labels=5,
                n_expected_labels=5,
            ),
        ]
        return self.generate_tests(data)


class TreeTestData(TestData):
    def raise_nonpositive_length_test_data(self):
        valid_splits = (Split({3, 4}, {0, 1, 2}), Split({0, 1}, {2, 3, 4}))
        data = [
            dict(splits=valid_splits, invalid_lengths=[0, 3]),
            dict(splits=valid_splits, invalid_lengths=[-2, 3]),
        ]
        return self.generate_tests(data)

    def raise_incompatible_size_test_data(self):
        data = [
            dict(
                splits=(Split({3, 4}, {0, 1, 2}), Split({0, 1}, {2, 3, 4})),
                lengths=[1, 2, 3],
            ),
        ]
        return self.generate_tests(data)

    def valid_tree_attributes_test_data(self):
        data = [
            dict(
                splits=(Split({3, 4}, {0, 1, 2}), Split({0, 1}, {2, 3, 4})),
                lengths=[2, 4],
            ),
            dict(splits=(), lengths=(), n_labels=5),
        ]

        return self.generate_tests(data)


class BHVMetric5TestData(TestData):
    def _get_owen_trees(self):
        # see Owen, 2011, fig2

        e1 = Split((1, 2), (0, 3, 4))
        e2 = Split((0, 1, 2), (3, 4))
        e3 = Split((1, 2, 3), (0, 4))

        tree_1 = Tree([e1, e2], [1.0, 1.0])
        tree_1_prime = Tree([e1, e2], [0.5, 1.0])

        tree_2 = Tree([e1, e3], [1.0, 1.5])
        tree_3 = Tree([e1], [1.0])

        return tree_1, tree_1_prime, tree_2, tree_3

    def _get_abby_trees(self):
        # https://plewis.github.io/applets/bhvspace/
        top_quartile_topology = (Split({3, 4}, {0, 1, 2}), Split({0, 1}, {2, 3, 4}))
        left_quartile_topology = (Split({2, 3}, {0, 1, 4}), Split({0, 4}, {3, 1, 2}))
        right_quartile_topology = (Split({2, 3}, {0, 1, 4}), Split({0, 1}, {2, 3, 4}))
        right_stratum_topology = (Split({0, 1}, {2, 3, 4}),)
        lengths = gs.array([3, 1])

        top_tree = Tree(top_quartile_topology, lengths)
        top_tree_shifted = Tree(top_quartile_topology, lengths + 1)
        right_tree = Tree(right_quartile_topology, lengths)
        mirrored_left_tree = Tree(left_quartile_topology, lengths)
        halfway_tree_right = Tree(right_stratum_topology, [1])

        return (
            top_tree,
            top_tree_shifted,
            right_tree,
            mirrored_left_tree,
            halfway_tree_right,
        )

    def dist_test_data(self):
        tree_1, tree_1_prime, tree_2, tree_3 = self._get_owen_trees()
        top_tree, top_tree_shifted, right_tree, mirrored_left_tree, _ = (
            self._get_abby_trees()
        )

        data = [
            dict(point_a=tree_1, point_b=tree_1_prime, t=0.5, expected=0.5),
            dict(point_a=tree_1, point_b=tree_2, expected=2.5),
            dict(point_a=tree_1, point_b=tree_3, expected=1.0),
            dict(point_a=tree_2, point_b=tree_3, expected=1.5),
            # distance same quadrant
            dict(point_a=top_tree, point_b=top_tree_shifted, expected=2**0.5),
            # distance different quadrant, vertical line
            dict(point_a=top_tree, point_b=right_tree, expected=6.0),
            # distance different quadrant, diagonal line
            dict(
                point_a=top_tree_shifted, point_b=right_tree, expected=(1 + 7**2) ** 0.5
            ),
            # distance different quadrant, geodesic through origin
            dict(
                point_a=top_tree,
                point_b=mirrored_left_tree,
                expected=2 * (1 + 3**2) ** 0.5,
            ),
        ]

        return self.generate_tests(data)

    def geodesic_test_data(self):
        tree_1, tree_1_prime, tree_2, tree_3 = self._get_owen_trees()
        (
            top_tree,
            top_tree_shifted,
            right_tree,
            mirrored_left_tree,
            halfway_tree_right,
        ) = self._get_abby_trees()
        star_tree = Tree((), [], n_labels=5)

        # check geodesic in same orthant
        tree_1_half = Tree(
            tree_1.topology.splits, (tree_1.lengths + tree_1_prime.lengths) / 2.0
        )

        data = [
            dict(
                initial_point=tree_1,
                end_point=tree_1_prime,
                t=0.5,
                expected=[tree_1_half],
            ),
            dict(initial_point=tree_1, end_point=tree_2, t=0.4, expected=[tree_3]),
            # geodesic same quadrant
            dict(
                initial_point=top_tree,
                end_point=top_tree_shifted,
                t=gs.array([0.0, 0.5, 1.0]),
                expected=[
                    top_tree,
                    Tree(top_tree.topology.splits, top_tree.lengths + 0.5),
                    top_tree_shifted,
                ],
            ),
            # geodesic different quadrant
            dict(
                initial_point=top_tree,
                end_point=right_tree,
                t=gs.array([0.0, 0.5, 1.0]),
                expected=[
                    top_tree,
                    halfway_tree_right,
                    right_tree,
                ],
            ),
            # geodesic different quadrant, geodesic through origin
            dict(
                initial_point=top_tree,
                end_point=mirrored_left_tree,
                t=gs.array([0.0, 0.3, 0.5, 0.7, 1.0]),
                expected=[
                    top_tree,
                    Tree(top_tree.topology.splits, [1.2, 0.4]),
                    star_tree,
                    Tree(mirrored_left_tree.topology.splits, [1.2, 0.4]),
                    mirrored_left_tree,
                ],
            ),
        ]

        return self.generate_tests(data)
