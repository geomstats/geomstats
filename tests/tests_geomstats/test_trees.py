""" Test file for the classes Split and Structure in trees.py.

Lead author: Jonas Lueg
"""
import geomstats.tests

from geomstats.geometry.trees import Split, Structure


class TestSplit(geomstats.tests.TestCase):
    def setup_method(self):
        """ Setup method. """
        pass

    def test_restr(self):
        """ Tests the attribute restr of class Split. """
        split1 = Split(n=5, part1=[2, 3], part2=[0, 1, 4])
        subset1 = {0, 1, 2}
        result = split1.restr(subset=subset1)
        expected = Split(n=5, part1=[0, 1], part2=[2])
        self.assertEqual(result, expected)

        split2 = Split(n=7, part1=[0, 1, 2, 3, 4, 5], part2=[6])
        subset2 = {0}
        result = split2.restr(subset=subset2)
        expected = Split(n=7, part1=[], part2=[0])
        self.assertEqual(result, expected)

    def test_contains(self):
        """ Tests the attribute contains of class Split. """
        split1 = Split(n=5, part1=[0, 4], part2=[1, 2, 3])
        subset1 = {0, 2}
        result = split1.contains(subset=subset1)
        expected = False
        self.assertEqual(result, expected)

        split2 = Split(n=10, part1=[0, 1, 2, 3, 6, 7, 8, 9], part2=[4, 5])
        subset2 = {0, 1, 2}
        result = split2.contains(subset=subset2)
        expected = True
        self.assertEqual(result, expected)

    def test_separates(self):
        """ Tests the attribute separates of class Split. """
        split1 = Split(n=3, part1=[0, 1], part2=[2])
        u1 = [0, 1]
        v1 = [2]
        result = split1.separates(u=u1, v=v1)
        expected = True
        self.assertEqual(result, expected)

        split2 = Split(n=5, part1=[0, 1, 2], part2=[3, 4])
        u2 = [0, 1, 2]
        v2 = [2, 3, 4]
        result = split2.separates(u=u2, v=v2)
        expected = False
        self.assertEqual(result, expected)

        split3 = Split(n=4, part1=[0, 1], part2=[2, 3])
        u3 = 1
        v3 = 3
        result = split3.separates(u=u3, v=v3)
        expected = True
        self.assertEqual(result, expected)

        split4 = Split(n=5, part1=[], part2=[0, 1, 2, 3, 4])
        u4 = 4
        v4 = 1
        result = split4.separates(u=u4, v=v4)
        expected = False
        self.assertEqual(result, expected)

    def test_point_to_split(self):
        """ Tests the attribute point_to_split of class Split. """
        split1a = Split(n=5, part1=[0, 4], part2=[1, 2, 3])
        split1b = Split(n=5, part1=[2, 3], part2=[0, 1, 4])

        result = split1a.point_to_split(split1b)
        expected = (1, 2, 3)
        self.assertEqual(result, expected)

        result = split1b.point_to_split(split1a)
        expected = (0, 1, 4)
        self.assertEqual(result, expected)

    def test_point_away_split(self):
        """ Tests the attribute point_away_split of class Split. """
        split1a = Split(n=5, part1=[0, 4], part2=[1, 2, 3])
        split1b = Split(n=5, part1=[2, 3], part2=[0, 1, 4])

        result = split1a.point_away_split(split1b)
        expected = (0, 4)
        self.assertEqual(result, expected)

        result = split1b.point_away_split(split1a)
        expected = (2, 3)
        self.assertEqual(result, expected)

    def test_compatible_with(self):
        """ Tests the attribute compatible_with of class Split. """
        split1a = Split(n=5, part1=[0, 4], part2=[1, 2, 3])
        split1b = Split(n=5, part1=[2, 3], part2=[0, 1, 4])

        result = split1a.compatible_with(split1b)
        expected = True
        self.assertEqual(result, expected)

        result = split1b.compatible_with(split1a)
        expected = True
        self.assertEqual(result, expected)

    def test_hash(self):
        """ Tests the attribute __hash__ of class Split. """
        split1a = Split(n=5, part1=[0, 4], part2=[1, 2, 3])
        split1b = Split(n=5, part1=[2, 3], part2=[0, 1, 4])

        result = hash(split1a) == hash(split1b)
        expected = False
        self.assertEqual(result, expected)

        split2a = Split(n=5, part1=[0], part2=[1, 2, 3])
        split2b = Split(n=4, part1=[0], part2=[1, 2, 3])
        result = hash(split2a) == hash(split2b)
        expected = False
        self.assertEqual(result, expected)

        split3a = Split(n=5, part1=[2, 1], part2=[0, 3, 4])
        split3b = Split(n=5, part1=[0, 4, 3], part2=[1, 2])
        result = hash(split3a) == hash(split3b)
        expected = True
        self.assertEqual(result, expected)


class TestStructure(geomstats.tests.TestCase):
    def setup_method(self):
        pass

    def test_partition(self):
        """ Tests the attribute partition of class Structure. """
        st1a = Structure(n=3, partition=((1, 0), (2,)), split_sets=((), ()))
        st1b = Structure(n=3, partition=((2,), (0, 1)), split_sets=((), ()))
        result = st1a.partition == st1b.partition
        expected = True
        self.assertEqual(result, expected)

        st2a = Structure(n=3, partition=((1,), (0,), (2,)), split_sets=((), (), ()))
        st2b = Structure(n=3, partition=((0,), (1,), (2,)), split_sets=((), (), ()))
        result = st2a.partition == st2b.partition
        expected = True
        self.assertEqual(result, expected)

    def test_partial_ordering(self):
        """ Tests the attributes __gt__, __ge__, __eq__, __lt__, __le__, __ne__. """
        sp1 = [[((0,), (1,))]]
        split_sets1 = [[Split(2, a, b) for a, b in splits] for splits in sp1]
        st1 = Structure(n=2, partition=((0, 1),), split_sets=split_sets1)
        st2 = Structure(n=2, partition=((0, 1),), split_sets=((),))
        result = [st1 > st2, st1 >= st2, st1 == st2, st1 < st2, st1 <= st2,
                  st1 != st2]
        expected = [True, True, False, False, False, True]
        self.assertEqual(result, expected)

        sp1 = [[((0,), (1, 2, 3)), ((3,), (0, 1, 2)), ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)), ((1, 2), (0, 3))]]
        split_sets1 = [[Split(4, a, b) for a, b in splits] for splits in sp1]
        st1 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(4, a, b) for a, b in splits] for splits in sp2]
        st2 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets2)

        result = [st1 > st2, st1 >= st2, st1 == st2, st1 < st2, st1 <= st2,
                  st1 != st2]
        expected = [True, True, False, False, False, True]
        self.assertEqual(result, expected)

        sp1 = [[((0,), (1, 2, 3)), ((3,), (0, 1, 2)), ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)), ((0, 2), (1, 3))]]
        split_sets1 = [[Split(4, a, b) for a, b in splits] for splits in sp1]
        st1 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(4, a, b) for a, b in splits] for splits in sp2]
        st2 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets2)

        result = [st1 > st2, st1 >= st2, st1 == st2, st1 < st2, st1 <= st2,
                  st1 != st2]
        expected = [False, False, False, False, False, True]
        self.assertEqual(result, expected)

        sp1 = [[((0,), (1, 2, 3)), ((3,), (0, 1, 2)), ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)), ((0, 2), (1, 3))]]
        split_sets1 = [[Split(4, a, b) for a, b in splits] for splits in sp1]
        st1 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        sp2 = [[((0,), (1, 2, 3)), ((3,), (0, 1, 2)), ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)), ((0, 3), (1, 2))]]
        split_sets2 = [[Split(4, a, b) for a, b in splits] for splits in sp2]
        st2 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets2)

        result = [st1 > st2, st1 >= st2, st1 == st2, st1 < st2, st1 <= st2,
                  st1 != st2]
        expected = [False, False, False, False, False, True]
        self.assertEqual(result, expected)

        sp1 = [[((0,), (1, 2, 3)), ((3,), (0, 1, 2)), ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3))]]
        split_sets1 = [[Split(4, a, b) for a, b in splits] for splits in sp1]
        st1 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(4, a, b) for a, b in splits] for splits in sp2]
        st2 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets2)

        result = [st1 > st2, st1 >= st2, st1 == st2, st1 < st2, st1 <= st2,
                  st1 != st2]
        expected = [False, False, False, False, False, True]
        self.assertEqual(result, expected)
