import pytest

import geomstats.backend as gs
from geomstats.metric_geometry.bhv_space import Tree, TreeTopology
from geomstats.test.test_case import TestCase
from geomstats.test_cases.metric_geometry.point_set import PointSetMetricTestCase


class TreeTopologyTestCase(TestCase):
    def _assert_raises(self, invalid_splits, expected_error_message):
        with pytest.raises(ValueError, match=expected_error_message):
            TreeTopology(invalid_splits)

    def test_raises_empty_splits(self, invalid_splits):
        """Check empty splits raise an error.

        Parameters
        ----------
        invalid_splits : list[Split]
            Splits containing an empty split.
        """
        self._assert_raises(invalid_splits, "Empty splits like .* are not allowed.")

    def test_raises_singleton(self, invalid_splits):
        """Check singleton splits raise an error.

        Parameters
        ----------
        invalid_splits : list[Split]
            Splits containing a singleton split.
        """
        self._assert_raises(
            invalid_splits,
            "Pendant edges / singleton splits like .* are not allowed.",
        )

    def test_valid_topology_attributes(self, splits, n_expected_labels, n_labels=None):
        """Check attributes of a valid topology.

        Parameters
        ----------
        splits : list[Split]
            Splits of the topology.
        n_expected_labels : int
            Expected number of labels.
        n_labels : int
            Number of labels.
        """
        tree_top = TreeTopology(splits, n_labels=n_labels)

        self.assertEqual(tree_top.splits, splits)
        self.assertEqual(tree_top.n_labels, n_expected_labels)


class TreeTestCase(TestCase):
    def _assert_raises(self, splits, lengths, expected_error_message):
        with pytest.raises(ValueError, match=expected_error_message):
            Tree(splits, lengths)

    def test_raise_nonpositive_length(self, splits, invalid_lengths):
        """Check non-positive lengths raise an error.

        Parameters
        ----------
        splits : list[Split]
            Splits of the tree.
        invalid_lengths : array-like
            Lengths with a non-positive entry.
        """
        self._assert_raises(splits, invalid_lengths, "Lengths must be positive.")

    def test_raise_incompatible_size(self, splits, lengths):
        """Check splits and lengths of different sizes raise an error.

        Parameters
        ----------
        splits : list[Split]
            Splits of the tree.
        lengths : array-like
            Lengths of a different size.
        """
        self._assert_raises(splits, lengths, "Splits and lengths different size. .*")

    def test_valid_tree_attributes(self, splits, lengths, n_labels=None):
        """Check attributes of a valid tree.

        Parameters
        ----------
        splits : list[Split]
            Splits of the tree.
        lengths : array-like
            Lengths of the splits.
        n_labels : int
            Number of labels.
        """
        tree = Tree(splits, lengths, n_labels=n_labels)

        self.assertEqual(len(tree.lengths), len(lengths))


class BHVMetricTestCase(PointSetMetricTestCase):
    def test_geodesic(self, initial_point, end_point, t, expected, atol=gs.atol):
        """Check geodesic.

        Parameters
        ----------
        initial_point : Tree
            Initial point of the geodesic.
        end_point : Tree
            End point of the geodesic.
        t : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        expected : list[Tree]
            Expected points of the geodesic.
        atol : float
            Absolute tolerance.
        """
        geod_func = self.space.metric.geodesic(initial_point, end_point)
        geod_points = geod_func(t)

        for geod_point, expected_point in zip(geod_points, expected):
            self.assertTrue(geod_point.equal(expected_point, atol))

    def test_raise_geodesic_out_bound(self, initial_point, end_point):
        """Check evaluating a geodesic outside [0, 1] raises an error.

        Parameters
        ----------
        initial_point : Tree or TreeBatch
            Initial point of the geodesic.
        end_point : Tree or TreeBatch
            End point of the geodesic.
        """
        geodesic = self.space.metric.geodesic(
            initial_point=initial_point, end_point=end_point
        )

        expected_error_message = "Geodesics only exist for 0<=t<=1.*"
        with pytest.raises(ValueError, match=expected_error_message):
            geodesic(-1)

        with pytest.raises(ValueError, match=expected_error_message):
            geodesic(1.1)
