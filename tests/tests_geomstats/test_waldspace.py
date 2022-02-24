import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.trees import Split, Structure, Wald
from geomstats.geometry.waldspace import WaldSpace


class TestSplit(geomstats.tests.TestCase):
    def test_restr(self):
        """Test the attribute restr of class Split."""
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
        """Test the attribute contains of class Split."""
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
        """Test the attribute separates of class Split."""
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
        """Test the attribute point_to_split of class Split."""
        split1a = Split(n=5, part1=[0, 4], part2=[1, 2, 3])
        split1b = Split(n=5, part1=[2, 3], part2=[0, 1, 4])

        result = split1a.point_to_split(split1b)
        expected = (1, 2, 3)
        self.assertEqual(result, expected)

        result = split1b.point_to_split(split1a)
        expected = (0, 1, 4)
        self.assertEqual(result, expected)

    def test_point_away_split(self):
        """Test the attribute point_away_split of class Split."""
        split1a = Split(n=5, part1=[0, 4], part2=[1, 2, 3])
        split1b = Split(n=5, part1=[2, 3], part2=[0, 1, 4])

        result = split1a.point_away_split(split1b)
        expected = (0, 4)
        self.assertEqual(result, expected)

        result = split1b.point_away_split(split1a)
        expected = (2, 3)
        self.assertEqual(result, expected)

    def test_compatible_with(self):
        """Test the attribute compatible_with of class Split."""
        split1a = Split(n=5, part1=[0, 4], part2=[1, 2, 3])
        split1b = Split(n=5, part1=[2, 3], part2=[0, 1, 4])

        result = split1a.compatible_with(split1b)
        expected = True
        self.assertEqual(result, expected)

        result = split1b.compatible_with(split1a)
        expected = True
        self.assertEqual(result, expected)

    def test_hash(self):
        """Test the attribute __hash__ of class Split."""
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
    def test_partition(self):
        """Test the attribute partition of class Structure."""
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
        """Test the attributes __gt__, __ge__, __eq__, __lt__, __le__, __ne__."""
        sp1 = [[((0,), (1,))]]
        split_sets1 = [[Split(2, a, b) for a, b in splits] for splits in sp1]
        st1 = Structure(n=2, partition=((0, 1),), split_sets=split_sets1)
        st2 = Structure(n=2, partition=((0, 1),), split_sets=((),))
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
        split_sets1 = [[Split(4, a, b) for a, b in splits] for splits in sp1]
        st1 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(4, a, b) for a, b in splits] for splits in sp2]
        st2 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets2)

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
        split_sets1 = [[Split(4, a, b) for a, b in splits] for splits in sp1]
        st1 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(4, a, b) for a, b in splits] for splits in sp2]
        st2 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets2)

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
        split_sets1 = [[Split(4, a, b) for a, b in splits] for splits in sp1]
        st1 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        sp2 = [
            [
                ((0,), (1, 2, 3)),
                ((3,), (0, 1, 2)),
                ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)),
                ((0, 3), (1, 2)),
            ]
        ]
        split_sets2 = [[Split(4, a, b) for a, b in splits] for splits in sp2]
        st2 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets2)

        result = [st1 > st2, st1 >= st2, st1 == st2, st1 < st2, st1 <= st2, st1 != st2]
        expected = [False, False, False, False, False, True]
        self.assertEqual(result, expected)

        sp1 = [
            [((0,), (1, 2, 3)), ((3,), (0, 1, 2)), ((1,), (0, 2, 3)), ((2,), (0, 1, 3))]
        ]
        split_sets1 = [[Split(4, a, b) for a, b in splits] for splits in sp1]
        st1 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(4, a, b) for a, b in splits] for splits in sp2]
        st2 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets2)

        result = [st1 > st2, st1 >= st2, st1 == st2, st1 < st2, st1 <= st2, st1 != st2]
        expected = [False, False, False, False, False, True]
        self.assertEqual(result, expected)


class TestWaldSpace(geomstats.tests.TestCase):
    """Class testing the methods and attributes of ``WaldSpace``."""

    def setup_method(self):
        gs.random.seed(1234)
        self.ws3 = WaldSpace(n=3)
        self.ws4 = WaldSpace(n=4)

    def test_belongs(self):
        """Test belongs method."""
        sp1 = [
            [
                ((0,), (1, 2, 3)),
                ((3,), (0, 1, 2)),
                ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)),
                ((1, 2), (0, 3)),
            ]
        ]
        split_sets1 = [[Split(4, a, b) for a, b in splits] for splits in sp1]
        st1 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        x1 = gs.array([0.1, 0.99, 0.81, 0.4, 0.01])
        wald = Wald(n=4, st=st1, x=x1).corr
        result = self.ws4.belongs(point=wald)
        expected = True
        self.assertAllClose(result, expected)

        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(4, a, b) for a, b in splits] for splits in sp2]
        st2 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets2)
        x2 = gs.array([0.01, 0.99])
        wald = Wald(n=4, st=st2, x=x2).corr
        result = self.ws4.belongs(point=wald)
        expected = True
        self.assertAllClose(result, expected)

        sp3 = [[((3,), (0, 1, 2)), ((1,), (0, 2, 3)), ((0, 2), (1, 3))]]
        split_sets3 = [[Split(4, a, b) for a, b in splits] for splits in sp3]
        st3 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets3)
        x3 = gs.array([0.1, 0.1, 0.1])
        wald = Wald(n=4, st=st3, x=x3).corr
        result = self.ws4.belongs(point=wald)
        expected = False
        self.assertAllClose(result, expected)

        sp4 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets4 = [[Split(4, a, b) for a, b in splits] for splits in sp4]
        st4 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets4)
        x4 = gs.array([-0.01, 0.98])
        wald = Wald(n=4, st=st4, x=x4).corr
        result = self.ws4.belongs(point=wald)
        expected = False
        self.assertAllClose(result, expected)

        sp5 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets5 = [[Split(4, a, b) for a, b in splits] for splits in sp5]
        st5 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets5)
        x5 = gs.array([0.7, 1.01])
        wald = Wald(n=4, st=st5, x=x5).corr
        result = self.ws4.belongs(point=wald)
        expected = False
        self.assertAllClose(result, expected)

        sp6 = [[], [], [], []]
        split_sets6 = [[Split(4, a, b) for a, b in splits] for splits in sp6]
        st6 = Structure(n=4, partition=((0,), (1,), (2,), (3,)), split_sets=split_sets6)
        x6 = gs.array([0])
        wald = Wald(n=4, st=st6, x=x6).corr
        result = self.ws4.belongs(point=wald)
        expected = True
        self.assertAllClose(result, expected)

    def test_belongs_vectorization(self):
        """Test belongs with several input points."""
        sp1 = [
            [
                ((0,), (1, 2, 3)),
                ((3,), (0, 1, 2)),
                ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)),
                ((1, 2), (0, 3)),
            ]
        ]
        split_sets1 = [[Split(4, a, b) for a, b in splits] for splits in sp1]
        st1 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets1)
        x1 = gs.array([0.1, 0.99, 0.81, 0.4, 0.01])
        wald1 = Wald(n=4, st=st1, x=x1).corr

        sp2 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets2 = [[Split(4, a, b) for a, b in splits] for splits in sp2]
        st2 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets2)
        x2 = gs.array([0.01, 0.99])
        wald2 = Wald(n=4, st=st2, x=x2).corr

        sp3 = [[((3,), (0, 1, 2)), ((1,), (0, 2, 3)), ((0, 2), (1, 3))]]
        split_sets3 = [[Split(4, a, b) for a, b in splits] for splits in sp3]
        st3 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets3)
        x3 = gs.array([0.1, 0.1, 0.1])
        wald3 = Wald(n=4, st=st3, x=x3).corr

        sp4 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets4 = [[Split(4, a, b) for a, b in splits] for splits in sp4]
        st4 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets4)
        x4 = gs.array([-0.01, 0.98])
        wald4 = Wald(n=4, st=st4, x=x4).corr

        sp5 = [[((1,), (2,))], [((0,), (3,))]]
        split_sets5 = [[Split(4, a, b) for a, b in splits] for splits in sp5]
        st5 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets5)
        x5 = gs.array([0.7, 1.01])
        wald5 = Wald(n=4, st=st5, x=x5).corr

        sp6 = [[], [], [], []]
        split_sets6 = [[Split(4, a, b) for a, b in splits] for splits in sp6]
        st6 = Structure(n=4, partition=((0,), (1,), (2,), (3,)), split_sets=split_sets6)
        x6 = gs.array([0])
        wald6 = Wald(n=4, st=st6, x=x6).corr

        point = gs.array([wald1, wald2, wald3, wald4, wald5, wald6])
        result = self.ws4.belongs(point=point)
        expected = gs.array([True, True, False, False, False, True])
        self.assertAllClose(result, expected)

    def test_random_point_and_belongs(self):
        """Test of random_point and belongs methods."""
        wald = self.ws4.random_point()
        p = wald
        if not self.ws4.a.belongs(p):
            print("alert spd")
        for i in range(self.ws4.n):
            if p[i, i] != 1:
                print("alert diag")
        import itertools as it

        for i, j, k in it.combinations(range(self.ws4.n), 3):
            if p[i, j] < p[i, k] * p[j, k]:
                print("alert triangles")
        for i, j, k, l in it.combinations(range(self.ws4.n), 4):
            if p[i, j] * p[k, l] < min(p[i, k] * p[j, l], p[i, l] * p[j, k]):
                print("alert four-point")
                print(f"{i}, {j}, {k}, {l}")
                print(f"{p[i, j]*p[k, l]}, {min(p[i, k]*p[j, l], p[i, l]*p[j, k])}")
        result = self.ws4.belongs(point=wald, atol=gs.atol)
        expected = True
        self.assertEqual(result, expected)

    def test_random_point_and_belongs_vectorization(self):
        """Test of random_point and belongs methods, vectorized version."""
        walds = self.ws4.random_point(n_samples=10)
        result = self.ws4.belongs(point=walds, atol=gs.atol)
        expected = gs.array([True] * 10)
        self.assertAllClose(result, expected, atol=gs.atol)
