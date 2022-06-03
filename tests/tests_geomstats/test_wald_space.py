from geomstats.geometry.stratified.wald_space import Split, Topology, Wald
from tests.conftest import Parametrizer, TestCase, np_backend
from tests.data.wald_space_data import SplitTestData, WaldSpaceTestData, WaldTestData
from tests.stratified_test_cases import PointSetTestCase, PointTestCase

IS_NOT_NP = not np_backend()


class TestWaldSpace(PointSetTestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP

    testing_data = WaldSpaceTestData()


class TestWald(PointTestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP

    _Point = Wald
    testing_data = WaldTestData()

    def test_generate_wald_belongs(self, point_args):
        generated_point = self._Point.generate_wald(*point_args)
        result = isinstance(generated_point, self._Point)
        self.assertAllClose(result, True)


class TestSplit(TestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP
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


class TestTopology(TestCase):
    """Class for testing the class Topology."""

    skip_all = IS_NOT_NP

    def test_partition(self):
        """Test the attribute partition of class Topology."""
        st1a = Topology(n=3, partition=((1, 0), (2,)), split_sets=((), ()))
        st1b = Topology(n=3, partition=((2,), (0, 1)), split_sets=((), ()))
        result = st1a.partition == st1b.partition
        expected = True
        self.assertEqual(result, expected)

        st2a = Topology(n=3, partition=((1,), (0,), (2,)), split_sets=((), (), ()))
        st2b = Topology(n=3, partition=((0,), (1,), (2,)), split_sets=((), (), ()))
        result = st2a.partition == st2b.partition
        expected = True
        self.assertEqual(result, expected)

    def test_partial_ordering(self):
        """Test the attributes __gt__, __ge__, __eq__, __lt__, __le__, __ne__."""
        sp1 = [[((0,), (1,))]]
        split_sets1 = [[Split(part1=a, part2=b) for a, b in splits] for splits in sp1]
        st1 = Topology(n=2, partition=((0, 1),), split_sets=split_sets1)
        st2 = Topology(n=2, partition=((0, 1),), split_sets=((),))
        result = [st1 > st2, st1 >= st2, st1 == st2, st1 < st2, st1 <= st2, st1 != st2]
        expected = [True, True, False, False, False, True]
        self.assertEqual(result, expected)

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
        st1 = Topology(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(a, b) for a, b in splits] for splits in sp2]
        st2 = Topology(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets2)

        result = [st1 > st2, st1 >= st2, st1 == st2, st1 < st2, st1 <= st2, st1 != st2]
        expected = [True, True, False, False, False, True]
        self.assertEqual(result, expected)

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
        st1 = Topology(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(part1=a, part2=b) for a, b in splits] for splits in sp2]
        st2 = Topology(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets2)

        result = [st1 > st2, st1 >= st2, st1 == st2, st1 < st2, st1 <= st2, st1 != st2]
        expected = [False, False, False, False, False, True]
        self.assertEqual(result, expected)

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
        st1 = Topology(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
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
        st2 = Topology(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets2)

        result = [st1 > st2, st1 >= st2, st1 == st2, st1 < st2, st1 <= st2, st1 != st2]
        expected = [False, False, False, False, False, True]
        self.assertEqual(result, expected)

        sp1 = [
            [((0,), (1, 2, 3)), ((3,), (0, 1, 2)), ((1,), (0, 2, 3)), ((2,), (0, 1, 3))]
        ]
        split_sets1 = [[Split(part1=a, part2=b) for a, b in splits] for splits in sp1]
        st1 = Topology(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(part1=a, part2=b) for a, b in splits] for splits in sp2]
        st2 = Topology(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets2)

        result = [st1 > st2, st1 >= st2, st1 == st2, st1 < st2, st1 <= st2, st1 != st2]
        expected = [False, False, False, False, False, True]
        self.assertEqual(result, expected)
