from geomstats.geometry.stratified.bhv_space import Split, Tree
from geomstats.test.data import TestData


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

    def dist_test_data(self):
        tree_1, tree_1_prime, tree_2, tree_3 = self._get_owen_trees()

        data = [
            dict(
                point_a=tree_1,
                point_b=tree_1_prime,
                t=0.5,
                expected=0.5,
            ),
            dict(point_a=tree_1, point_b=tree_2, expected=2.5),
            dict(point_a=tree_1, point_b=tree_3, expected=1.0),
            dict(point_a=tree_2, point_b=tree_3, expected=1.5),
        ]

        return self.generate_tests(data)

    def geodesic_test_data(self):
        tree_1, tree_1_prime, tree_2, tree_3 = self._get_owen_trees()

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
        ]

        return self.generate_tests(data)
