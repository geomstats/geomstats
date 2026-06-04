import random

import pytest

from geomstats.metric_geometry.bhv_space import Tree, TreeSpace, TreeTopology
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import TestCase, np_only
from geomstats.test_cases.metric_geometry.point_set import (
    PointSetMetricTestCase,
    PointSetTestCase,
    PointTestCase,
)

from .data.bhv_space import (
    BHVMetric5TestData,
    BHVMetricTestData,
    TreeTestData,
    TreeTopologyTestData,
)
from .data.point_set import PointSetTestData, PointTestData


class TestTreeTopology(TestCase, metaclass=DataBasedParametrizer):
    testing_data = TreeTopologyTestData()

    def test_raises_error(
        self,
        invalid_splits,
        expected_error_message,
    ):
        with pytest.raises(ValueError, match=expected_error_message):
            TreeTopology(invalid_splits)

    def test_raises_empty_splits(self, invalid_splits):
        expected_error_message = "Empty splits like .* are not allowed."
        self.test_raises_error(invalid_splits, expected_error_message)

    def test_raises_singleton(self, invalid_splits):
        expected_error_message = (
            "Pendant edges / singleton splits like .* are not allowed."
        )
        self.test_raises_error(invalid_splits, expected_error_message)

    def test_valid_topology_attributes(self, splits, n_expected_labels, n_labels=None):
        tree_top = TreeTopology(splits, n_labels=n_labels)

        self.assertEqual(tree_top.splits, splits)
        self.assertEqual(tree_top.n_labels, n_expected_labels)


class TestTree(TestCase, metaclass=DataBasedParametrizer):
    testing_data = TreeTestData()

    def test_raises_error(self, splits, lengths, expected_error_message):
        with pytest.raises(ValueError, match=expected_error_message):
            Tree(splits, lengths)

    def test_raise_nonpositive_length(self, splits, invalid_lengths):
        expected_error_message = "Lengths must be positive."

        self.test_raises_error(splits, invalid_lengths, expected_error_message)

    def test_raise_incompatible_size(self, splits, lengths):
        expected_error_message = "Splits and lengths different size. .*"
        self.test_raises_error(splits, lengths, expected_error_message)

    def test_valid_tree_attributes(self, splits, lengths, n_labels=None):
        tree = Tree(splits, lengths, n_labels=n_labels)

        self.assertEqual(len(tree.lengths), len(lengths))


class TestTreeAsPoint(PointTestCase, metaclass=DataBasedParametrizer):
    _n_labels = random.randint(4, 5)
    space = TreeSpace(n_labels=_n_labels, equip=False)

    testing_data = PointTestData()


class TestTreeSpace(PointSetTestCase, metaclass=DataBasedParametrizer):
    _n_labels = random.randint(4, 5)
    space = TreeSpace(n_labels=_n_labels, equip=False)

    testing_data = PointSetTestData()


@np_only
class TestBHVMetric(PointSetMetricTestCase, metaclass=DataBasedParametrizer):
    _n_labels = random.randint(4, 5)
    space = TreeSpace(n_labels=_n_labels, equip=True)

    testing_data = BHVMetricTestData()

    @pytest.mark.random
    def test_raise_geodesic_out_bound(self, n_points):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        geodesic = self.space.metric.geodesic(
            initial_point=initial_point, end_point=end_point
        )

        expected_error_message = "Geodesics only exist for 0<=t<=1.*"
        with pytest.raises(ValueError, match=expected_error_message):
            geodesic(-1)

        with pytest.raises(ValueError, match=expected_error_message):
            geodesic(1.1)


@pytest.mark.smoke
@np_only
class TestBHVMetric5(PointSetMetricTestCase, metaclass=DataBasedParametrizer):
    space = TreeSpace(n_labels=5, equip=True)

    testing_data = BHVMetric5TestData()

    def test_geodesic(self, initial_point, end_point, t, expected, atol):
        geod_func = self.space.metric.geodesic(initial_point, end_point)
        geod_points = geod_func(t)

        for geod_point, expected_point in zip(geod_points, expected):
            self.assertTrue(geod_point.equal(expected_point, atol))
