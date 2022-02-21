import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry.waldspace import WaldSpace
from geomstats.geometry.trees import Structure, Split, Wald


class TestWaldSpace(geomstats.tests.TestCase):
    """Class testing the methods of MyManifold.

    In the class TestMyManifold, each test method:
    - needs to start with `test_`
    - represents a unit-test, i.e. it tests one and only one method
    or attribute of the class MyManifold,
    - ends with the line: `self.assertAllClose(result, expected)`,
    as in the examples below.
    """

    def setup_method(self):
        gs.random.seed(1234)
        self.ws3 = WaldSpace(n=3)
        self.ws4 = WaldSpace(n=4)

    def test_belongs(self):
        """ Test belongs method. """
        sp1 = [[((0,), (1, 2, 3)), ((3,), (0, 1, 2)), ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)), ((1, 2), (0, 3))]]
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
        x6 = gs.array([])
        wald = Wald(n=4, st=st6, x=x6).corr
        result = self.ws4.belongs(point=wald)
        expected = True
        self.assertAllClose(result, expected)

    def test_belongs_vectorization(self):
        """ Test belongs with several input points. """
        sp1 = [[((0,), (1, 2, 3)), ((3,), (0, 1, 2)), ((1,), (0, 2, 3)),
                ((2,), (0, 1, 3)), ((1, 2), (0, 3))]]
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
        x6 = gs.array([])
        wald6 = Wald(n=4, st=st6, x=x6).corr

        point = gs.array([wald1, wald2, wald3, wald4, wald5, wald6])
        result = self.ws4.belongs(point=point)
        expected = gs.array([True, True, False, False, False, True])
        self.assertAllClose(result, expected)

    def test_random_point_and_belongs(self):
        """ Test of random_point and belongs methods. """
        wald = self.ws4.random_point()
        result = self.ws4.belongs(point=wald)
        expected = True
        self.assertAllClose(result, expected)

    def test_random_point_and_belongs_vectorization(self):
        """ Test of random_point and belongs methods, vectorized version. """
        walds = self.ws4.random_point(n_samples=10)
        p = walds[5]
        if not self.ws4.a.belongs(p):
            print("alert spd")
        for i in range(self.ws4.n):
            if p[i, i] != 1:
                print("alert diag")
        import itertools as it
        for i, j, k in it.combinations(range(self.ws4.n), 3):
            if p[i, j] < p[i, k]*p[j, k]:
                print("alert triangles")
        for i, j, k, l in it.combinations(range(self.ws4.n), 4):
            if p[i, j]*p[k, l] < min(p[i, k]*p[j, l], p[i, l]*p[j, k]):
                print("alert four-point")
                print(f"{i}, {j}, {k}, {l}")
                print(f"{p[i, j]*p[k, l]}, {min(p[i, k]*p[j, l], p[i, l]*p[j, k])}")
        result = self.ws4.belongs(point=walds)
        expected = gs.array([True] * 10)
        self.assertAllClose(result, expected)

    # def test_lift(self):
    #     """ Test of the lift method. """
    #     sp1 = [[((1,), (2,))], [((0,), (3,))]]
    #     split_sets1 = [[Split(4, a, b) for a, b in splits] for splits in sp1]
    #     st1 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets1)
    #     x1 = gs.array([0.2, 0.4])
    #     wald1 = Wald(n=4, st=st1, x=x1)
    #     result = wald1.corr
    #     expected = gs.array([[1., 0., 0., 0.8],
    #                          [0., 1., 0.6, 0.],
    #                          [0., 0.6, 1., 0.],
    #                          [0.8, 0., 0., 1.]])
    #     self.assertAllClose(result, expected)
    #
    #     sp2 = [[((0,), (1, 2, 3)), ((3,), (0, 1, 2)), ((1,), (0, 2, 3)),
    #             ((2,), (0, 1, 3)), ((1, 2), (0, 3))]]
    #     split_sets2 = [[Split(4, a, b) for a, b in splits] for splits in sp2]
    #     st2 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets2)
    #     x2 = gs.array([0.1, 0.2, 0.4, 0.4, 0.1])
    #     wald2 = Wald(n=4, st=st2, x=x2)
    #     result = self.ws4.lift(point=wald2)
    #     expected = gs.array([[1., 0.432, 0.324, 0.81],
    #                          [0.432, 1., 0.48, 0.432],
    #                          [0.324, 0.48, 1., 0.324],
    #                          [0.81, 0.432, 0.324, 1.]])
    #     self.assertAllClose(result, expected)
    #
    #     sp3 = [[], [], [], []]
    #     split_sets3 = [[Split(4, a, b) for a, b in splits] for splits in sp3]
    #     st3 = Structure(n=4, partition=((0,), (1,), (2,), (3,)), split_sets=split_sets3)
    #     x3 = gs.array([])
    #     wald3 = Wald(n=4, st=st3, x=x3)
    #     result = self.ws4.lift(point=wald3)
    #     expected = gs.eye(4)
    #     self.assertAllClose(result, expected)
    #
    # def test_lift_vectorization(self):
    #     """ Test of the lift method, vectorized version. """
    #     sp1 = [[((1,), (2,))], [((0,), (3,))]]
    #     split_sets1 = [[Split(4, a, b) for a, b in splits] for splits in sp1]
    #     st1 = Structure(n=4, partition=((1, 2), (0, 3)), split_sets=split_sets1)
    #     x1 = gs.array([0.2, 0.4])
    #     wald1 = Wald(n=4, st=st1, x=x1)
    #
    #     sp2 = [[((0,), (1, 2, 3)), ((3,), (0, 1, 2)), ((1,), (0, 2, 3)),
    #             ((2,), (0, 1, 3)), ((1, 2), (0, 3))]]
    #     split_sets2 = [[Split(4, a, b) for a, b in splits] for splits in sp2]
    #     st2 = Structure(n=4, partition=((0, 1, 2, 3),), split_sets=split_sets2)
    #     x2 = gs.array([0.1, 0.2, 0.4, 0.4, 0.1])
    #     wald2 = Wald(n=4, st=st2, x=x2)
    #
    #     sp3 = [[], [], [], []]
    #     split_sets3 = [[Split(4, a, b) for a, b in splits] for splits in sp3]
    #     st3 = Structure(n=4, partition=((0,), (1,), (2,), (3,)), split_sets=split_sets3)
    #     x3 = gs.array([])
    #     wald3 = Wald(n=4, st=st3, x=x3)
    #
    #     result = self.ws4.lift(point=gs.array([wald1, wald2, wald3]))
    #     expected = gs.array([gs.array([[1., 0., 0., 0.8],
    #                                    [0., 1., 0.6, 0.],
    #                                    [0., 0.6, 1., 0.],
    #                                    [0.8, 0., 0., 1.]]),
    #                          gs.array([[1., 0.432, 0.324, 0.81],
    #                                    [0.432, 1., 0.48, 0.432],
    #                                    [0.324, 0.48, 1., 0.324],
    #                                    [0.81, 0.432, 0.324, 1.]]),
    #                          gs.eye(4)])
    #     self.assertAllClose(result, expected)
