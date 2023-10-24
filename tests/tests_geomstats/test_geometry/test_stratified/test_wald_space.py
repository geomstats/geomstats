from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.stratified.point_set import (
    PointSetTestCase,
    PointTestCase,
)

from .data.wald_space import (
    SplitTestData,
    TopologyTestData,
    WaldSpaceTestData,
    WaldTestData,
)


class TestWaldSpace(PointSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = WaldSpaceTestData()


class TestWald(PointTestCase, metaclass=DataBasedParametrizer):
    testing_data = WaldTestData()
    _Point = testing_data._Point

    def test_generate_wald_belongs(self, point_args):
        generated_point = self._Point.generate_wald(*point_args)
        result = isinstance(generated_point, self._Point)
        self.assertAllClose(result, True)


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


class TestTopology(TestCase, metaclass=DataBasedParametrizer):
    testing_data = TopologyTestData()

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
