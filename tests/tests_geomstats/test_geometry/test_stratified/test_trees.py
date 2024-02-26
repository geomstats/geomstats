from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import TestCase

from .data.trees import BaseTopologyTestData, SplitTestData


class TestSplit(TestCase, metaclass=DataBasedParametrizer):
    testing_data = SplitTestData()

    def test_restrict_to(self, split, subset, expected):
        result = split.restrict_to(subset=subset)
        self.assertEqual(result, expected)

    def test_part_contains(self, split, subset, expected):
        result = split.part_contains(subset=subset)
        self.assertEqual(result, expected)

    def test_separates(self, split, u, v, expected):
        result = split.separates(u=u, v=v)
        self.assertEqual(result, expected)

    def test_get_part_towards(self, split_a, split_b, expected):
        result = split_a.get_part_towards(split_b)
        self.assertEqual(result, expected)

    def test_get_part_away_from(self, split_a, split_b, expected):
        result = split_a.get_part_away_from(split_b)
        self.assertEqual(result, expected)

    def test_is_compatible(self, split_a, split_b, expected):
        result = split_a.is_compatible(split_b)
        self.assertEqual(result, expected)

        result = split_b.is_compatible(split_a)
        self.assertEqual(result, expected)

    def test_hash(self, split_a, split_b, expected):
        result = hash(split_a) == hash(split_b)
        self.assertEqual(result, expected)


class TestBaseTopology(TestCase, metaclass=DataBasedParametrizer):
    testing_data = BaseTopologyTestData()

    def test_partition(self, st_a, st_b, expected):
        result = st_a.partition == st_b.partition
        self.assertEqual(result, expected)

    def test_partial_ordering(self, st_a, st_b, expected):
        """Test the attributes __gt__, __ge__, __eq__, __lt__, __le__, __ne__."""
        result = [
            st_a > st_b,
            st_a >= st_b,
            st_a == st_b,
            st_a < st_b,
            st_a <= st_b,
            st_a != st_b,
        ]
        self.assertEqual(result, expected)
