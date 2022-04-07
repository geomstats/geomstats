import geomstats.tests
from geomstats.stratified_geometry.wald_space import Split, Topology


class TestSplit(geomstats.tests.TestCase):
    def test_restrict_to(self):
        """Test the attribute restrict_to of class Split."""
        split1 = Split(part1=[2, 3], part2=[0, 1, 4])
        subset1 = {0, 1, 2}
        result = split1.restrict_to(subset=subset1)
        expected = Split(part1=[0, 1], part2=[2])
        self.assertEqual(result, expected)

        split2 = Split(part1=[0, 1, 2, 3, 4, 5], part2=[6])
        subset2 = {0}
        result = split2.restrict_to(subset=subset2)
        expected = Split(part1=[], part2=[0])
        self.assertEqual(result, expected)

    def test_part_contains(self):
        """Test the attribute part_contains of class Split."""
        split1 = Split(part1=[0, 4], part2=[1, 2, 3])
        subset1 = {0, 2}
        result = split1.part_contains(subset=subset1)
        expected = False
        self.assertEqual(result, expected)

        split2 = Split(part1=[0, 1, 2, 3, 6, 7, 8, 9], part2=[4, 5])
        subset2 = {0, 1, 2}
        result = split2.part_contains(subset=subset2)
        expected = True
        self.assertEqual(result, expected)

    def test_separates(self):
        """Test the attribute separates of class Split."""
        split1 = Split(part1=[0, 1], part2=[2])
        u1 = [0, 1]
        v1 = [2]
        result = split1.separates(u=u1, v=v1)
        expected = True
        self.assertEqual(result, expected)

        split2 = Split(part1=[0, 1, 2], part2=[3, 4])
        u2 = [0, 1, 2]
        v2 = [2, 3, 4]
        result = split2.separates(u=u2, v=v2)
        expected = False
        self.assertEqual(result, expected)

        split3 = Split(part1=[0, 1], part2=[2, 3])
        u3 = 1
        v3 = 3
        result = split3.separates(u=u3, v=v3)
        expected = True
        self.assertEqual(result, expected)

        split4 = Split(part1=[], part2=[0, 1, 2, 3, 4])
        u4 = 4
        v4 = 1
        result = split4.separates(u=u4, v=v4)
        expected = False
        self.assertEqual(result, expected)

    def test_get_part_towards(self):
        """Test the attribute get_part_towards of class Split."""
        split1a = Split(part1=[0, 4], part2=[1, 2, 3])
        split1b = Split(part1=[2, 3], part2=[0, 1, 4])

        result = split1a.get_part_towards(split1b)
        expected = (1, 2, 3)
        self.assertEqual(result, expected)

        result = split1b.get_part_towards(split1a)
        expected = (0, 1, 4)
        self.assertEqual(result, expected)

    def test_get_part_away_from(self):
        """Test the attribute get_part_away_from of class Split."""
        split1a = Split(part1=[0, 4], part2=[1, 2, 3])
        split1b = Split(part1=[2, 3], part2=[0, 1, 4])

        result = split1a.get_part_away_from(split1b)
        expected = (0, 4)
        self.assertEqual(result, expected)

        result = split1b.get_part_away_from(split1a)
        expected = (2, 3)
        self.assertEqual(result, expected)

    def test_is_compatible(self):
        """Test the attribute is_compatible of class Split."""
        split1a = Split(part1=[0, 4], part2=[1, 2, 3])
        split1b = Split(part1=[2, 3], part2=[0, 1, 4])

        result = split1a.is_compatible(split1b)
        expected = True
        self.assertEqual(result, expected)

        result = split1b.is_compatible(split1a)
        expected = True
        self.assertEqual(result, expected)

    def test_hash(self):
        """Test the attribute __hash__ of class Split."""
        split1a = Split(part1=[0, 4], part2=[1, 2, 3])
        split1b = Split(part1=[2, 3], part2=[0, 1, 4])
        result = hash(split1a) == hash(split1b)
        expected = False
        self.assertEqual(result, expected)

        split2a = Split(part1=[0], part2=[1, 2, 3])
        split2b = Split(part1=[0], part2=[1, 3, 2])
        result = hash(split2a) == hash(split2b)
        expected = True
        self.assertEqual(result, expected)

        split3a = Split(part1=[2, 1], part2=[0, 3, 4])
        split3b = Split(part1=[0, 4, 3], part2=[1, 2])
        result = hash(split3a) == hash(split3b)
        expected = True
        self.assertEqual(result, expected)


class TestTopology(geomstats.tests.TestCase):
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
