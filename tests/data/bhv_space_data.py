import random

import geomstats.backend as gs
from geomstats.geometry.stratified.bhv_space import BHVMetric, Split, Tree, TreeSpace
from tests.data_generation import (
    _PointMetricTestData,
    _PointSetTestData,
    _PointTestData,
)


def _get_example_tree():
    n_labels = 3
    splits = (((0, 1), (2,)), ((0, 2), (1,)), ((0,), (1, 2)))
    splits = [Split(a, b) for a, b in splits]

    w = gs.array([0.1, 0.2, 0.3])
    lengths = gs.abs(gs.log(1 - w))

    corr = gs.abs(
        gs.log(gs.array([[1.0, 0.56, 0.63], [0.56, 1.0, 0.72], [0.63, 0.72, 1.0]]))
    )

    return n_labels, splits, lengths, corr


class TreeTestData(_PointTestData):
    _Point = Tree

    def to_array_test_data(self):
        n_labels, splits, lengths, expected_corr = _get_example_tree()

        smoke_data = [
            dict(point_args=(n_labels, splits, lengths), expected=expected_corr),
        ]

        return self.generate_tests(smoke_data)


class TreeSpaceTestData(_PointSetTestData):
    _Point = Tree
    _PointSet = TreeSpace

    n_samples = 2
    n_points_list = [1] + random.sample(range(2, 5), n_samples - 1)
    n_labels_list = random.sample(range(2, 4), n_samples)
    space_args_list = [(n_labels,) for n_labels in n_labels_list]

    def belongs_test_data(self):
        n_labels, splits, lengths, _ = _get_example_tree()
        point = Tree(n_labels, splits, lengths)

        smoke_data = [
            dict(space_args=(n_labels,), points=point, expected=True),
            dict(space_args=(n_labels - 1,), points=point, expected=False),
        ]

        return self.generate_tests(smoke_data)

    def set_to_array_test_data(self):
        n_labels, splits, lengths, corr_ = _get_example_tree()
        point = Tree(n_labels, splits, lengths)

        smoke_data = [
            dict(
                space_args=(n_labels,),
                points=[point, point],
                expected=gs.stack([corr_, corr_]),
            )
        ]

        return self.generate_tests(smoke_data)


class BHVMetricTestData(_PointMetricTestData):

    _Point = Tree
    _PointSet = TreeSpace
    _PointSetMetric = BHVMetric

    n_samples = 2
    n_labels_list = random.sample(range(2, 4), n_samples)
    space_args_list = [(n_labels,) for n_labels in n_labels_list]

    def _get_owen_trees(self):
        # see Owen, 2011, fig2

        n_labels = 5

        e1 = Split((1, 2), (0, 3, 4))
        e2 = Split((0, 1, 2), (3, 4))
        e3 = Split((1, 2, 3), (0, 4))

        tree_1 = Tree(n_labels, [e1, e2], [1.0, 1.0])
        tree_1_prime = Tree(n_labels, [e1, e2], [0.5, 1.0])

        tree_2 = Tree(n_labels, [e1, e3], [1.0, 1.5])
        tree_3 = Tree(n_labels, [e1], [1.0])

        return tree_1, tree_1_prime, tree_2, tree_3

    def dist_test_data(self):
        tree_1, tree_1_prime, tree_2, tree_3 = self._get_owen_trees()
        n_labels = tree_1.n_labels

        smoke_data = [
            dict(
                space_args=(n_labels,),
                point_a=tree_1,
                point_b=tree_1_prime,
                t=0.5,
                expected=0.5,
            ),
            dict(space_args=(n_labels,), point_a=tree_1, point_b=tree_2, expected=2.5),
            dict(space_args=(n_labels,), point_a=tree_1, point_b=tree_3, expected=1.0),
            dict(space_args=(n_labels,), point_a=tree_2, point_b=tree_3, expected=1.5),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_test_data(self):
        tree_1, tree_1_prime, tree_2, tree_3 = self._get_owen_trees()
        n_labels = tree_1.n_labels

        # check geodesic in same orthant
        tree_1_half = Tree(
            n_labels, tree_1.splits, (tree_1.lengths + tree_1_prime.lengths) / 2.0
        )

        smoke_data = [
            dict(
                space_args=(n_labels,),
                start_point=tree_1,
                end_point=tree_1_prime,
                t=0.5,
                expected=gs.reshape(gs.abs(gs.log(tree_1_half.corr)), (1, 5, 5)),
            ),
            dict(
                space_args=(n_labels,),
                start_point=tree_1,
                end_point=tree_2,
                t=0.4,
                expected=gs.reshape(gs.abs(gs.log(tree_3.corr)), (1, 5, 5)),
            ),
        ]

        return self.generate_tests(smoke_data)
