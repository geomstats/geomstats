"""Class for the BHV Tree Space.

Lead author: Jonas Lueg

References
----------
.. [BHV01] Billera, L. J., S. P. Holmes, K. Vogtmann.
    "Geometry of the Space of Phylogenetic Trees."
    Advances in Applied Mathematics,
    volume 27, issue 4, pages 733-767, 2001.
    https://doi.org/10.1006%2Faama.2001.0759
.. [OP11] Owen, M., J. S. Provan.
    "A Fast Algorithm for Computing Geodesic Distances in Tree Space."
    IEEE/ACM Transactions on Computational Biology and Bioinformatics,
    volume 8, issue 1, pages 2-13, 2011.
    https://doi.org/10.1109/TCBB.2010.3
"""

import itertools as it

import networkx as nx
import numpy as np

import geomstats.backend as gs
from geomstats.geometry.stratified.point_set import (
    Point,
    PointBatch,
    PointSet,
    PointSetMetric,
)
from geomstats.geometry.stratified.trees import (
    ForestTopology,
    Split,
    delete_splits,
    generate_splits,
)
from geomstats.geometry.stratified.vectorization import broadcast_lists, vectorize_point


class TreeTopology(ForestTopology):
    r"""The topology of a tree with only interior edges, using a split-based representation.

    Parameters
    ----------
    splits : list[Split]
        The structure of the tree in form of a set of splits of the set of labels.
        Pendant edges (i.e., singleton splits) are not permitted.
    n_labels : int, optional
        Number of labels, the set of labels is then :math:`\{0,\dots,n-1\}`.
        Needed to instantiate star tree at origin, as labels cannot be inferred from empty split set.

    Attributes
    ----------
    n_labels : int
        Number of labels, the set of labels is then :math:`\{0,\dots,n-1\}`.
    where : dict
        Give the index of a split in the flattened list of all splits.
    sep : list of int
        An increasing list of numbers between 0 and m, where m is the total number
        of splits in ``self.split_sets``, starting with 0, where each number
        indicates that a new connected component starts at that index.
        Useful for example for unraveling the tuple of all splits into
        ``self.split_sets``.
    paths : list of dict
        A list of dictionaries, each dictionary is for the respective connected
        component of the forest, and the items of each dictionary are for each pair
        of labels u, v, u < v in the respective component, a list of the splits on the
        unique path between the labels u and v.
    support : list of array-like
        For each split, give an :math:`n\times n` dimensional matrix, where the
        uv-th entry is ``True`` if the split separates the labels u and v, else
        ``False``.
    """

    def __init__(self, splits, n_labels=None):
        if not splits:
            super().__init__(
                partition=(set(range(n_labels)),),
                split_sets=(splits,),
            )
        else:
            TreeTopology._check_no_pendant_edges(splits)
            super().__init__(
                partition=(tuple(splits[0].part1.union(splits[0].part2)),),
                split_sets=(splits,),
            )

    @staticmethod
    def _check_no_pendant_edges(splits):
        """Verify that there are no pendant edges (i.e., singleton splits).

        Parameters
        ----------
        splits : list[Split]
            The structure of the tree in form of a set of splits of the set of labels.
        """
        for split in splits:
            if len(split.part1) == 1 or len(split.part2) == 1:
                raise ValueError(
                    f"Pendant edges / singleton splits like {split} are not allowed."
                )

    @property
    def splits(self):
        """Splits.

        Returns
        -------
        splits : list[Split]
            The structure of the tree in form of a set of splits of the set of labels.
        """
        return self.split_sets[0]

    @property
    def labels(self):
        """Node labels.

        Returns
        -------
        labels : tuple[int]
            Node labels.
        """
        return self.partition[0]


class Tree(Point):
    r"""A class for unrooted phylogenetic trees, which are elements of the BHV space.

    An unrooted phylogenetic tree is represented by a list of splits and a vector of their corresponding lengths.
    Only interior edges are permitted; a tree with N leaves has 2^N - N - 2 possible interior edge splits.
    Pendant edges can be represented by taking the cartesian product of BHV space with Euclidean space.
    Rooted trees with N leaves can be represented as unrooted trees with N+1 leaves, assigning the root to 0.

    Parameters
    ----------
    splits : list[Split]
        The structure of the tree in form of a set of splits of the set of labels.
    lengths : array-like, shape=[n_splits]
        The edge lengths of the splits, a vector containing positive numbers.
    n_labels : int, optional
        Number of labels, the set of labels is then :math:`\{0,\dots,n-1\}`.
        Needed to instantiate star tree at origin, as labels cannot be inferred from empty split set.


    Attributes
    ----------
    topology : TreeTopology
        The topology of the tree.
    lengths : array-like, shape=[n_splits]
        The edge lengths of the splits, a vector containing positive numbers.
    """

    def __init__(self, splits, lengths, n_labels=None):
        Tree._check_valid_lengths(lengths, len(splits))

        self.topology = TreeTopology(splits=splits, n_labels=n_labels)
        self.lengths = gs.array(
            [
                length
                for _, length in sorted(
                    zip(splits, lengths), key=lambda x: self.topology.where.get(x[0])
                )
            ]
        )

    def __repr__(self):
        """Return the string representation of the tree.

        This string representation requires that one can retrieve all necessary
        information from the string.

        Returns
        -------
        string_of_tree : str
            Return the string representation of the tree.
        """
        return repr((self.topology.splits, tuple(self.lengths)))

    def __str__(self):
        """Return the fancy printable string representation of the tree.

        This string representation does NOT require that one can retrieve all necessary
        information from the string, but this string representation is required to be
        readable by a human.

        Returns
        -------
        string_of_tree : str
            Return the fancy readable string representation of the tree.
        """
        return f"({self.topology};{str(self.lengths)})"

    def _equal_single(self, point, atol=gs.atol):
        """Check equality against another point.

        Parameters
        ----------
        point : Tree
            Point to compare against point.
        atol : float

        Returns
        -------
        is_equal : bool
        """
        if self.topology != point.topology:
            return False

        return gs.all(gs.abs(self.lengths - point.lengths) < atol)

    @staticmethod
    def _check_valid_lengths(lengths, n_splits):
        """Verify edge lengths are in (0, infinity).

        Parameters
        ----------
        lengths : array-like, shape=[n_splits]
            The edge lengths of the splits, a vector containing positive numbers.
        len_splits : int
            Number of splits.
        """
        if len(lengths) != n_splits:
            raise ValueError(
                f"Splits and lengths different size. {len(lengths)} != {n_splits}"
            )

        for length in lengths:
            if length <= 0:
                raise ValueError(f"Lengths must be positive. {length} is not allowed.")

    @vectorize_point((1, "point"))
    def equal(self, point, atol=gs.atol):
        """Check equality against another point.

        Parameters
        ----------
        point : Tree or TreeBatch
            Point to compare against point.
        atol : float

        Returns
        -------
        is_equal : array-like, shape=[...]
        """
        return gs.array([self._equal_single(point_, atol) for point_ in point])


class TreeBatch(PointBatch):
    """Tree batch."""

    @property
    def topology(self):
        """Tree topology.

        Returns
        -------
        topology : list[TreeTopology]
        """
        return [point.topology for point in self]

    @property
    def lengths(self):
        """Edge lengths.

        Returns
        -------
        lengths : array-like, shape=[n_points, n_splits]
        """
        return gs.array([point.lengths for point in self])


class TreeSpace(PointSet):
    """Class for the Tree space, a point set containing phylogenetic trees.

    A topological space. Points in Tree space are instances of the class :class:`Tree`:
    phylogenetic trees with edge lengths between 0 and infinity.
    For the space of trees see also [BHV01]_.

    Parameters
    ----------
    n_labels : int
        The number of labels in the trees.
    """

    def __init__(self, n_labels, equip=True):
        if n_labels < 4:
            raise ValueError(
                f"BHV space only defined for N >= 4. You tried {n_labels}."
            )
        self.n_labels = n_labels
        super().__init__(equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return BHVMetric

    def _belongs_single(self, point, atol=gs.atol):
        """Check if a point belongs to Tree space.

        Parameters
        ----------
        point : Tree
            The point to be checked.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : bool
            Boolean denoting if point belongs to Tree space.
        """
        if not isinstance(point, Tree) or (point.topology.n_labels != self.n_labels):
            return False
        return gs.all(point.lengths > -atol)

    @vectorize_point((1, "point"))
    def belongs(self, point, atol=gs.atol):
        """Check if a point belongs to Tree space.

        Parameters
        ----------
        point : Tree or TreeBatch
            The point to be checked.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...]
            Boolean denoting if point belongs to Tree space.
        """
        return gs.array([self._belongs_single(point_, atol) for point_ in point])

    def _generate_random_tree(self, p_keep=0.9, btol=1e-8):
        r"""Generate a random instance of ``Tree``.

        Parameters
        ----------
        p_keep : float between 0 and 1
            The probability that a sampled edge is kept and not deleted randomly.
            To be precise, it is not exactly the probability, as some edges cannot be
            deleted since the requirement that two labels are separated by a split might
            be violated otherwise.
            Defaults to 0.9
        btol: float
            Tolerance for the boundary of the edge lengths. Defaults to 1e-08.
        """
        labels = list(range(self.n_labels))

        initial_splits = generate_splits(labels, exclude_singletons=True)
        splits = delete_splits(initial_splits, labels, p_keep, check=False)

        x = gs.random.uniform(size=(len(splits),), low=0, high=1)
        x = gs.minimum(gs.maximum(btol, x), 1 - btol)
        lengths = gs.maximum(btol, gs.abs(gs.log(1 - x)))

        return Tree(splits, lengths)

    def random_point(self, n_samples=1, p_keep=0.9, btol=1e-8):
        """Sample a random point in Tree space.

        Parameters
        ----------
        n_samples : int
            Number of samples. Defaults to 1.
        exclude_pendant_edges : bool
            Phylogenetic trees do not usually have lengths on pendant (external) edges (ie, those touching a leaf).
        p_keep : float between 0 and 1
            The probability that a sampled edge is kept and not deleted randomly.
            To be precise, it is not exactly the probability, as some edges cannot be
            deleted since the requirement that two labels are separated by a split might
            be violated otherwise.
            Defaults to 0.9
        btol: float
            Tolerance for the boundary of the edge lengths. Defaults to 1e-08.

        Returns
        -------
        samples : Tree or TreeBatch
            Points sampled in Tree space.
        """
        trees = [
            self._generate_random_tree(
                p_keep=p_keep,
                btol=btol,
            )
            for _ in range(n_samples)
        ]

        if n_samples == 1:
            return trees[0]

        return TreeBatch(trees)


class BHVMetric(PointSetMetric):
    """BHV metric for Tree Space for phylogenetic trees.

    The BHV Tree Space as it is introduced in [BHV01]_, a metric space that
    is CAT(0), and there exist unique geodesics between each pair of points
    in the BHV Space.
    The polynomial time algorithm for computing the distance and geodesic
    between two points is implemented, following the definitions and results
    of [OP11]_.
    There, computing the geodesic between two trees is called the 'Geodesic
    Tree Path' problem, and that is why some methods below (not visible to the
    user though) start with the letters 'gtp'.

    Parameters
    ----------
    total_space : TreeSpace
        Set with quotient structure.
    """

    def __init__(self, space):
        super().__init__(space=space)
        self.geodesic_solver = GTPSolver(n_labels=space.n_labels)

    def squared_dist(self, point_a, point_b):
        """Compute the squared distance between two points.

        Parameters
        ----------
        point_a : Tree or TreeBatch
            A point in BHV Space.
        point_b : Tree or TreeBatch
            A point in BHV Space.

        Returns
        -------
        squared_dist : array-like, shape=[...]
            The squared distance between the two points.
        """
        return self.geodesic_solver.squared_dist(point_a, point_b)

    def dist(self, point_a, point_b):
        """Compute the distance between two points.

        Parameters
        ----------
        point_a : Tree or TreeBatch
            A point in BHV Space.
        point_b : Tree or TreeBatch
            A point in BHV Space.

        Returns
        -------
        dist : array-like, shape=[...]
            The distance between the two points.
        """
        return self.geodesic_solver.dist(point_a, point_b)

    def geodesic(self, initial_point, end_point):
        """Compute the geodesic between two points.

        Parameters
        ----------
        initial_point : Tree or TreeBatch
            A point in BHV Space.
        end_point : Tree or TreeBatch
            A point in BHV Space.

        Returns
        -------
        geodesic : callable
            The geodesic between the two points. Takes parameter t, that is the time
            between 0 and 1 at which the corresponding point on the path is returned.
        """
        return self.geodesic_solver.geodesic(
            initial_point=initial_point, end_point=end_point
        )


class GTPSolver:
    """'Geodesic Tree Path' problem solver [OP11]_.

    Essentially uses Theorem 2.4 from [OP11]_.

    Parameters
    ----------
    tol : float
        Tolerance for the algorithm, in particular for the decision problem in the
        GTP algorithm in [OP11] to avoid unambiguity.
    """

    def __init__(self, n_labels, tol=1e-8):
        self.n_labels = n_labels
        self.tol = tol

    def _squared_dist_single(self, point_a, point_b):
        """Compute the squared distance between two points.

        Parameters
        ----------
        point_a : Tree
            A point in BHV Space.
        point_b : Tree
            A point in BHV Space.

        Returns
        -------
        squared_dist : array-like, shape=[...]
            The squared distance between the two points.
        """
        sp_a = dict(zip(point_a.topology.splits, point_a.lengths))
        sp_b = dict(zip(point_b.topology.splits, point_b.lengths))

        common_a, common_b, supports = self._trees_with_common_support(
            sp_a,
            sp_b,
        )
        sq_dist_common = sum((common_a[s] - common_b[s]) ** 2 for s in common_a.keys())
        sq_dist_parts = sum(
            (
                gs.sqrt(sum(sp_a[s] ** 2 for s in a))
                + gs.sqrt(sum(sp_b[s] ** 2 for s in b))
            )
            ** 2
            for supp_a, supp_b in supports.values()
            for a, b in zip(supp_a, supp_b)
        )

        return sq_dist_common + sq_dist_parts

    @vectorize_point((1, "point_a"), (2, "point_b"))
    def squared_dist(self, point_a, point_b):
        """Compute the squared distance between two points.

        Parameters
        ----------
        point_a : Tree or TreeBatch
            A point in BHV Space.
        point_b : Tree or TreeBatch
            A point in BHV Space.

        Returns
        -------
        squared_dist : array-like, shape=[...]
            The squared distance between the two points.
        """
        point_a, point_b = broadcast_lists(point_a, point_b)
        sq_dists = gs.array(
            [
                self._squared_dist_single(point_a_, point_b_)
                for point_a_, point_b_ in zip(point_a, point_b)
            ]
        )

        if len(sq_dists) == 1:
            return sq_dists[0]

        return sq_dists

    def dist(self, point_a, point_b):
        """Compute the distance between two points.

        Parameters
        ----------
        point_a : Tree or TreeBatch
            A point in BHV Space.
        point_b : Tree or TreeBatch
            A point in BHV Space.

        Returns
        -------
        dist : array-like, shape=[...]
            The distance between the two points.
        """
        return gs.sqrt(self.squared_dist(point_a, point_b))

    def _geodesic_single(self, initial_point, end_point):
        """Compute the geodesic between two points.

        Parameters
        ----------
        initial_point : Tree
            A point in BHV Space.
        end_point : Tree
            A point in BHV Space.

        Returns
        -------
        geodesic : callable
            The geodesic between the two points. Takes parameter t, that is the time
            between 0 and 1 at which the corresponding point on the path is returned.
        """
        sp_a = dict(zip(initial_point.topology.splits, initial_point.lengths))
        sp_b = dict(zip(end_point.topology.splits, end_point.lengths))
        common_a, common_b, supports = self._trees_with_common_support(
            sp_a,
            sp_b,
        )
        ratios = {
            part: [
                gs.sqrt(sum(sp_a[s] ** 2 for s in a) / sum(sp_b[s] ** 2 for s in b))
                for a, b in zip(supp_a, supp_b)
            ]
            for part, (supp_a, supp_b) in supports.items()
        }

        def geodesic_t(t):
            if (t < 0) or (t > 1):
                raise ValueError(f"Geodesics only exist for 0<=t<=1. You tried {t}.")

            if t == 0.0:
                return initial_point
            if t == 1.0:
                return end_point

            t_ratio = t / (1 - t)
            splits_t = {s: (1 - t) * common_a[s] + t * common_b[s] for s in common_a}

            for part, (supp_a, supp_b) in supports.items():
                index = gs.argmax([t_ratio <= _r for _r in ratios[part] + [np.inf]])
                splits_t_a = {
                    s: sp_a[s] * (1 - t - t / _r)
                    for a_k, _r in zip(supp_a[index:], ratios[part][index:])
                    for s in a_k
                }
                splits_t_b = {
                    s: sp_b[s] * (t - (1 - t) * _r)
                    for b_k, _r in zip(supp_b[:index], ratios[part][:index])
                    for s in b_k
                }
                splits_t = {**splits_t, **splits_t_a, **splits_t_b}

            splits_lengths = [
                (split, length)
                for split, length in splits_t.items()
                if length > self.tol
            ]

            if len(splits_lengths) == 0:
                return Tree((), [], n_labels=self.n_labels)
            return Tree(
                splits=[sl[0] for sl in splits_lengths],
                lengths=[sl[1] for sl in splits_lengths],
            )

        def geodesic_(t):
            if isinstance(t, (float, int)):
                t = gs.array([t])

            return TreeBatch([geodesic_t(t_) for t_ in t])

        return geodesic_

    @vectorize_point((1, "initial_point"), (2, "end_point"))
    def geodesic(self, initial_point, end_point):
        """Compute the geodesic between two points.

        Parameters
        ----------
        initial_point : Tree or TreeBatch
            A point in BHV Space.
        end_point : Tree or TreeBatch
            A point in BHV Space.

        Returns
        -------
        geodesic : callable
            The geodesic between the two points. Takes parameter t, that is the time
            between 0 and 1 at which the corresponding point on the path is returned.
        """
        initial_point, end_point = broadcast_lists(initial_point, end_point)

        def _vec(t, fncs):
            if len(fncs) == 1:
                return fncs[0](t)

            return [fnc(t) for fnc in fncs]

        fncs = [
            self._geodesic_single(initial_point_, end_point_)
            for initial_point_, end_point_ in zip(initial_point, end_point)
        ]

        return lambda t: _vec(t, fncs=fncs)

    def _trees_with_common_support(self, splits_a, splits_b):
        """Compute the support that corresponds to a geodesic for common split sets.

        We refer to the splits of the tree corresponding to splits_a as A,
        and B analogously.
        This method divides the split sets into smaller split sets that have distinct
        support and then use the method ``gtp_trees_with_distinct_support``.
        For each of these smaller subsets, return the support in a dictionary.

        The common splits are returned separately as well as the respective edge
        lengths. A split in A that is not in B but compatible with all
        splits of B is added to the common splits of B with length zero,
        and vice versa for splits in B.

        Parameters
        ----------
        splits_a : dict of Split, float
            The splits in A and their respective lengths.
        splits_b : dict of Split, float
            The splits in B and their respective lengths.

        Returns
        -------
        common_a : dict
            Containing the splits of A that are also in B, as well as the splits of B
            that are compatible with all splits in A, given edge length zero.
        common_b : dict
            Containing the splits of B that are also in A, as well as the splits of A
            that are compatible with all splits in B, given edge length zero.
        supports: dict
            Containing for each subtree the respective support.
        """
        pendants = {
            Split(part1=[i], part2=[j for j in range(self.n_labels) if j != i])
            for i in range(self.n_labels)
        }
        sp_a, sp_b = set(splits_a.keys()), set(splits_b.keys())
        common = sp_a & sp_b
        only_a = sp_a - common
        only_b = sp_b - common
        easy_a = {s for s in only_a if gs.all(list(map(s.is_compatible, only_b)))}
        easy_b = {s for s in only_b if gs.all(list(map(s.is_compatible, only_a)))}
        total_a = (sp_a | easy_b) - pendants
        total_b = (sp_b | easy_a) - pendants

        cut_splits = (common | easy_a | easy_b) - pendants

        trees_a = self._cut_tree_at_splits(total_a, cut_splits)
        trees_b = self._cut_tree_at_splits(total_b, cut_splits)
        supports = {
            part: self._trees_with_distinct_support(
                {s: splits_a[s] for s in trees_a[part]},
                {s: splits_b[s] for s in trees_b[part]},
            )
            for part in trees_a.keys()
            if trees_a[part] and trees_b[part]
        }
        common = common | easy_a | easy_b
        common_a = {s: splits_a[s] if s in sp_a else 0 for s in common}
        common_b = {s: splits_b[s] if s in sp_b else 0 for s in common}
        return common_a, common_b, supports

    def _cut_tree_at_splits(self, splits, cut_splits):
        """Cut a tree, given by splits, at all edges in cut_splits.

        Starting with the partition that consists of all labels and is assigned all
        splits,
        the tree is successively cut into parts by the splits in cut_splits.
        Accordingly, the set of labels is cut successively into parts and the set of all
        splits is also cut successively into the respective parts.

        Parameters
        ----------
        splits : iterable of split
            The tree given via its splits. Each split corresponds to an edge.
        cut_splits : iterable of Split
            A subset of splits, the edges at which the tree is cut.

        Returns
        -------
        partition : dict of tuple, tuple
            A dictionary, where the keys form a partition of the set of labels
            (0,...,n_labels-1),
            and each key is assigned the tuple of splits that are part of the subtree
            that
            the respective set of labels is spanning.
        """
        partition = {tuple(range(self.n_labels)): splits}
        for cut in cut_splits:
            try:
                labels, subtree = [
                    (_, subtree) for _, subtree in partition.items() if cut in subtree
                ][0]
            except IndexError:
                continue
            splits = set(subtree) - {cut}
            part1 = tuple(set(labels) & set(cut.part1))
            part2 = tuple(set(labels) & set(cut.part2))
            subtree1 = {s for s in splits if part1 == cut.get_part_towards(s)}
            subtree2 = splits - subtree1

            partition.pop(labels)
            partition = {
                **partition,
                tuple(part1): tuple(subtree1),
                tuple(part2): tuple(subtree2),
            }
        return partition

    def _trees_with_distinct_support(self, splits_a, splits_b):
        """Compute the support that corresponds to a geodesic for disjoint split sets.

        This is essentially the GTP algorithm from [1], starting with a cone path and
        iteratively updating the support, solving in each iteration an extension problem
        for
        each support pair.

        The Extension Problem gives a minimum cut of a graph and two-set partitions C1
        and
        C2 of A, and D1 and D2 of B, respectively. If the value of the minimum cut is
        greater or equal to one minus some tolerance, then the support pair (A,B) is
        split
        into (C1,D1) and (C2,D2).

        Parameters
        ----------
        splits_a : dict of Split, float
            The splits in A and their respective lengths.
        splits_b : dict of Split, float
            The splits in B and their respective lengths.

        Returns
        -------
        support_a : tuple of tuple
            The support partition of A corresponding to a geodesic.
        support_b : tuple of tuple
            The support partition of B corresponding to a geodesic.
        """
        old_support_a = (tuple(splits_a.keys()),)
        old_support_b = (tuple(splits_b.keys()),)
        weights_a = {split: splits_a[split] ** 2 for split in splits_a}
        weights_b = {split: splits_b[split] ** 2 for split in splits_b}
        while 1:
            new_support_a, new_support_b = (), ()
            for pair_a, pair_b in zip(old_support_a, old_support_b):
                pair_a_w = {s: weights_a[s] for s in pair_a}
                pair_b_w = {s: weights_b[s] for s in pair_b}
                value, c1, c2, d1, d2 = self._solve_extension_problem(
                    pair_a_w, pair_b_w
                )
                if value >= 1 - self.tol:
                    new_support_a += (pair_a,)
                    new_support_b += (pair_b,)
                else:
                    new_support_a += (c1, c2)
                    new_support_b += (d1, d2)
            if len(new_support_a) == len(old_support_a):
                return new_support_a, new_support_b
            old_support_a, old_support_b = new_support_a, new_support_b

    @staticmethod
    def _solve_extension_problem(sq_splits_a, sq_splits_b):
        """Solve the extension problem in [1] for sets of splits with squared weights.

        Solving the min weight vertex cover with respect to the incompatibility graph in
        the Extension Problem in [1] is equivalent to solving the minimum cut problem
        for
        the following directed graph with edges that have 'capacities'.
        The set of vertices are the splits in A, the splits in B, a sink and a source
        node.
        The source is connected to all splits in A, each edge has the normalized squared
        weight of the split it is attached to. Analogously, each split in B is connected
         to
        the sink and the corresponding edge has normalized squared weight of the split
        in B.
        Finally, each split in A is attached to a split in B whenever the splits are not
        compatible. The edge is given infinite capacity.

        The minimum cut returns the two-set partition (V, V_bar) of the set of vertices
        and
        its value, that is the sum of all capacities of edges from V to V_bar, such
        that the
        source is in V and the sink is in V_bar.

        If the value is larger or equal than one (possibly with respect to some
        tolerance),
        then a geodesic is found and there is no need to update anything.
        Else, the sets A and B are separated into sets
        C_1 = A intersection V_bar, C_2 = A intersection V,
        D_1 = B intersection V_bar, D_2 = B intersection V.
        Then, the new support is (i.e. A and B are replaced with) (C_1, C_2) and
        (D_1, D_2)
        (here, the notation from [1], GTP algorithm is used).

        Parameters
        ----------
        sq_splits_a : dict of Split, float
            Dictionary of splits in A with squared length associated to each split.
        sq_splits_b : dict of Split, float
            Dictionary of splits in B with squared length associated to each split.

        Returns
        -------
        value : float
            The value of the minimum cut.
        c1 : set of Split
            First part of A that it is split into.
        c2 : set of Split
            Second part of A that it is split into.
        d1 : set of Split
            First part of B that it is split into.
        d2 : set of Split
            Second part of B that it is split into.
        """
        total_a, total_b = sum(sq_splits_a.values()), sum(sq_splits_b.values())
        graph = nx.DiGraph()
        for split, weight in sq_splits_a.items():
            graph.add_edge("source", split, capacity=weight / total_a)
        for split, weight in sq_splits_b.items():
            graph.add_edge(split, "sink", capacity=weight / total_b)
        for split_a, split_b in it.product(sq_splits_a.keys(), sq_splits_b.keys()):
            if not split_a.is_compatible(split_b):
                graph.add_edge(split_a, split_b)

        min_value, (v, v_bar) = nx.minimum_cut(graph, "source", "sink")
        a = set(sq_splits_a.keys())
        b = set(sq_splits_b.keys())
        v = set(v)
        v_bar = set(v_bar)
        return min_value, tuple(a & v_bar), tuple(a & v), tuple(b & v_bar), tuple(b & v)
