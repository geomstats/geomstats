from geomstats.metric_geometry.trees import generate_splits
from geomstats.test.test_case import TestCase


class GenerateSplitsTestCase(TestCase):
    def test_len(self, n_labels, n_expected_splits, exclude_singletons=False):
        """Check number of generated splits.

        Parameters
        ----------
        n_labels : int
            Number of labels.
        n_expected_splits : int
            Expected number of splits.
        exclude_singletons : bool
            Whether to exclude splits with a singleton part.
        """
        labels = list(range(n_labels))
        splits = generate_splits(labels, exclude_singletons=exclude_singletons)

        self.assertEqual(len(splits), n_expected_splits)


class SplitTestCase(TestCase):
    def test_restrict_to(self, split, subset, expected):
        """Check restriction of a split to a subset.

        Parameters
        ----------
        split : Split
            Split.
        subset : set
            Subset of labels.
        expected : Split
            Expected restricted split.
        """
        result = split.restrict_to(subset=subset)
        self.assertEqual(result, expected)

    def test_part_contains(self, split, subset, expected):
        """Check whether a part of a split contains a subset.

        Parameters
        ----------
        split : Split
            Split.
        subset : set
            Subset of labels.
        expected : bool
            Expected result.
        """
        result = split.part_contains(subset=subset)
        self.assertEqual(result, expected)

    def test_separates(self, split, u, v, expected):
        """Check whether a split separates two labels.

        Parameters
        ----------
        split : Split
            Split.
        u : int or iterable
            Label or labels.
        v : int or iterable
            Label or labels.
        expected : bool
            Expected result.
        """
        result = split.separates(u=u, v=v)
        self.assertEqual(result, expected)

    def test_get_part_towards(self, split_a, split_b, expected):
        """Check part of a split towards another split.

        Parameters
        ----------
        split_a : Split
            Split.
        split_b : Split
            Split.
        expected : tuple
            Expected part of `split_a`.
        """
        result = split_a.get_part_towards(split_b)
        self.assertEqual(result, expected)

    def test_get_part_away_from(self, split_a, split_b, expected):
        """Check part of a split away from another split.

        Parameters
        ----------
        split_a : Split
            Split.
        split_b : Split
            Split.
        expected : tuple
            Expected part of `split_a`.
        """
        result = split_a.get_part_away_from(split_b)
        self.assertEqual(result, expected)

    def test_is_compatible(self, split_a, split_b, expected):
        """Check compatibility of two splits, in both orders.

        Parameters
        ----------
        split_a : Split
            Split.
        split_b : Split
            Split.
        expected : bool
            Expected result.
        """
        result = split_a.is_compatible(split_b)
        self.assertEqual(result, expected)

        result = split_b.is_compatible(split_a)
        self.assertEqual(result, expected)

    def test_hash(self, split_a, split_b, expected):
        """Check whether two splits have the same hash.

        Parameters
        ----------
        split_a : Split
            Split.
        split_b : Split
            Split.
        expected : bool
            Whether the hashes are expected to be equal.
        """
        result = hash(split_a) == hash(split_b)
        self.assertEqual(result, expected)


class BaseTopologyTestCase(TestCase):
    def test_partition(self, st_a, st_b, expected):
        """Check whether two topologies have the same partition.

        Parameters
        ----------
        st_a : BaseTopology
            Topology.
        st_b : BaseTopology
            Topology.
        expected : bool
            Whether the partitions are expected to be equal.
        """
        result = st_a.partition == st_b.partition
        self.assertEqual(result, expected)

    def test_partial_ordering(self, st_a, st_b, expected):
        """Check the partial ordering of topologies.

        Parameters
        ----------
        st_a : BaseTopology
            Topology.
        st_b : BaseTopology
            Topology.
        expected : list[bool]
            Expected results of ``>``, ``>=``, ``==``, ``<``, ``<=`` and
            ``!=``, in this order.
        """
        result = [
            st_a > st_b,
            st_a >= st_b,
            st_a == st_b,
            st_a < st_b,
            st_a <= st_b,
            st_a != st_b,
        ]
        self.assertEqual(result, expected)
