"""Class for the BHV Tree Space.

Class ``Tree``.
A tree is essentially a phylogenetic tree with edges having length greater than zero.
The representation of the tree is via splits, the edge lengths are stored in a vector.

Class ``TreeSpace``.
A topological space. Points in Tree space are instances of the class :class:`Tree`:
phylogenetic trees with edge lengths between 0 and infinity.
For the space of trees see also [BHV01].

Class ``BHVSpace``.
The BHV Tree Space as it is introduced in [BHV01], a metric space that is CAT(0), and
there exist unique geodesics between each pair of points in the BHV Space.
The polynomial time algorithm for computing the distance and geodesic between two points
is implemented, following the definitions and results of [OP11].
There, computing the geodesic between two trees is called the 'Geodesic Tree Path'
problem, and that is why some methods below (not visible to the user though) start with
the letters 'gtp'.

Lead author: Jonas Lueg

References
----------
[BHV01] Billera, L. J., S. P. Holmes, K. Vogtmann.
        "Geometry of the Space of Phylogenetic Trees."
        Advances in Applied Mathematics,
        volume 27, issue 4, pages 733-767, 2001.
        https://doi.org/10.1006%2Faama.2001.0759

[OP11]  Owen, M., J. S. Provan.
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
    PointSet,
    PointSetMetric,
    _vectorize_point,
)
from geomstats.geometry.stratified.wald_space import Split, Topology, Wald


class Tree(Wald, Point):
    r"""A class for trees, that are phylogenetic trees, elements of the BHV space.

    Parameters
    ----------
    n_labels : int
        Number of labels, the set of labels is then :math:`\{0,\dots,n_labels-1\}`.
    splits : list[Split]
        The structure of the tree in form of a set of splits of the set of labels.
    lengths : array-like
        The edge lengths of the splits, a vector containing positive numbers.
    """

    def __init__(self, n_labels, splits, lengths):
        top = Topology(
            n_labels=n_labels,
            partition=(tuple(i for i in range(n_labels)),),
            split_sets=(splits,),
        )
        self.lengths = gs.zeros(len(lengths))
        for s, val in zip(splits, lengths):
            self.lengths[top.where[s]] = val

        super().__init__(topology=top, weights=1 - gs.exp(-self.lengths))
        self.n = n_labels
        self.splits = self.topology.split_sets[0]

    def to_array(self):
        """Turn the tree into a numpy array, namely its distance matrix.

        Returns
        -------
        array_of_tree : array-like, shape=[n_labels, n_labels]
            The distance matrix corresponding to the wald.
        """
        return gs.abs(gs.log(self.corr))

    def __repr__(self):
        """Return the string representation of the tree.

        This string representation requires that one can retrieve all necessary
        information from the string.

        Returns
        -------
        string_of_tree : str
            Return the string representation of the tree.
        """
        return repr((self.splits, tuple(self.lengths)))

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


class TreeSpace(PointSet):
    """Class for the Tree space, a point set containing phylogenetic trees.

    Parameters
    ----------
    n_labels : int
        The number of labels in the trees.
    """

    def __init__(self, n_labels):
        super().__init__()
        self.n_labels = n_labels

    @_vectorize_point((1, "points"))
    def set_to_array(self, points):
        """Convert a set of points into an array.

        Parameters
        ----------
        points : list of Tree, shape=[...]
            Number of samples of trees to turn into an array.

        Returns
        -------
        points_array : array-like, shape=[...]
            Array of the trees that are turned into arrays.
        """
        results = gs.array([tree.to_array() for tree in points])
        return results

    @_vectorize_point((1, "point"))
    def belongs(self, point, atol):
        """Check if a point belongs to Tree space.

        Parameters
        ----------
        point : Tree or list of Tree
            The point to be checked.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : bool
            Boolean denoting if `point` belongs to Tree space.
        """
        belongs = [gs.all(tree.lengths > atol) for tree in point]
        return belongs

    def random_point(self, n_samples=1, p_keep=0.9, btol=1e-08):
        """Sample a random point in Tree space.

        Parameters
        ----------
        n_samples : int
            Number of samples. Defaults to 1.
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
        samples : Tree or list of Tree, shape=[n_samples]
            Points sampled in Tree space.
        """

        def tree_from_wald(wald_: Wald):
            """Construct a tree from a wald that is a tree.

            Parameters
            ----------
            wald_ : Wald
                The wald that is converted into a tree.

            Returns
            -------
            tree_ : Tree
                The resulting tree.
            """
            tree_ = Tree(
                n_labels=wald_.n_labels,
                splits=wald_.topology.split_sets[0],
                lengths=gs.maximum(btol, gs.abs(gs.log(1 - wald_.weights))),
            )
            return tree_

        if n_samples == 1:
            wald = Wald.generate_wald(
                self.n_labels, p_keep, p_new=1, btol=btol, check=False
            )
            return tree_from_wald(wald_=wald)
        waelder = [
            Wald.generate_wald(self.n_labels, p_keep, p_new=1, btol=btol, check=False)
            for _ in range(n_samples)
        ]
        samples = [tree_from_wald(wald_=wald) for wald in waelder]
        return samples


class BHVMetric(PointSetMetric):
    """BHV Tree Space for phylogenetic trees.

    Parameters
    ----------
    n_labels : int
        The number of labels.
    """

    def __init__(self, n_labels):
        super().__init__(space=TreeSpace(n_labels=n_labels))

    @property
    def n_labels(self):
        return self.space.n_labels

    @_vectorize_point((1, "point_a"), (2, "point_b"))
    def dist(self, point_a, point_b, tol=1e-8, squared=False):
        """Compute the distance between two points in BHV Space.

        Essentially uses Theorem 2.4 from [OP11].

        Parameters
        ----------
        point_a : Tree
            A point in BHV Space.
        point_b : Tree
            A point in BHV Space.
        tol : float
            Tolerance for the algorithm, in particular for the decision problem in the
            GTP algorithm in [OP11] to avoid unambiguity.
        squared : bool
            If true, return the squared distance.

        Returns
        -------
        dist : float
            The distance between the two points.
        """
        # TODO: split in `dist` and `squared_dist`
        return self._gtp_dist(point_a, point_b, tol, squared)

    # @_vectorize_point((1, "point_a"), (2, "point_b"))
    def geodesic(self, point_a, point_b, tol=10**-8):
        """Compute the geodesic between two points in BHV Space.

        Essentially uses Theorem 2.4 from [OP11].

        Parameters
        ----------
        point_a : Tree
            A point in BHV Space.
        point_b : Tree
            A point in BHV Space.
        tol : float
            Tolerance for the algorithm, in particular for the decision problem in the
            GTP algorithm in [OP11] to avoid unambiguity.

        Returns
        -------
        geodesic : callable
            The geodesic between the two points. Takes parameter t, that is the time
            between 0 and 1 at which the corresponding point on the path is returned.
        """
        return self._gtp_geodesic(point_a=point_a, point_b=point_b, tol=tol)

    def _gtp_dist(self, point_a, point_b, tol=10**-8, squared=False):
        """Compute the distance between two points in BHV Space.

        Essentially uses Theorem 2.4 from [OP11].

        Parameters
        ----------
        point_a : Tree
            A point in BHV Space.
        point_b : Tree
            A point in BHV Space.
        tol : float
            Tolerance for the algorithm, in particular for the decision problem in the
            GTP algorithm in [OP11] to avoid unambiguity.
        squared : bool
            If true, return the squared distance.

        Returns
        -------
        dist : float
            The distance between the two points.
        """
        sp_a = {split: length for split, length in zip(point_a.splits, point_a.lengths)}
        sp_b = {split: length for split, length in zip(point_b.splits, point_b.lengths)}
        common_a, common_b, supports = self._gtp_trees_with_common_support(
            sp_a,
            sp_b,
            tol,
        )
        sq_dist_common = sum((common_a[s] - common_b[s]) ** 2 for s in common_a.keys())
        sq_dist_parts = sum(
            (
                np.sqrt(sum(sp_a[s] ** 2 for s in a))
                + np.sqrt(sum(sp_b[s] ** 2 for s in b))
            )
            ** 2
            for supp_a, supp_b in supports.values()
            for a, b in zip(supp_a, supp_b)
        )
        if squared:
            return sq_dist_common + sq_dist_parts
        return np.sqrt(sq_dist_common + sq_dist_parts)

    def _gtp_geodesic(self, point_a, point_b, tol=1e-8):
        """Compute the geodesic between two points in BHV Space.

        Essentially uses Theorem 2.4 from [OP11].

        Parameters
        ----------
        point_a : Tree
            A point in BHV Space.
        point_b : Tree
            A point in BHV Space.
        tol : float
            Tolerance for the algorithm, in particular for the decision problem in the
            GTP algorithm in [OP11] to avoid unambiguity.

        Returns
        -------
        geodesic : callable
            The geodesic between the two points. Takes parameter t, that is the time
            between 0 and 1 at which the corresponding point on the path is returned.
        """
        sp_a = dict(zip(point_a.splits, point_a.lengths))
        sp_b = dict(zip(point_b.splits, point_b.lengths))
        common_a, common_b, supports = self._gtp_trees_with_common_support(
            sp_a,
            sp_b,
            tol,
        )
        ratios = {
            part: [
                np.sqrt(sum(sp_a[s] ** 2 for s in a) / sum(sp_b[s] ** 2 for s in b))
                for a, b in zip(supp_a, supp_b)
            ]
            for part, (supp_a, supp_b) in supports.items()
        }

        # @_vectorize_point((1, "t"))
        def geodesic_(t):
            if t == 0:
                return point_a
            elif t == 1:
                return point_b
            t_ratio = t / (1 - t)
            splits_t = {s: (1 - t) * common_a[s] + t * common_b[s] for s in common_a}
            for part, (supp_a, supp_b) in supports.items():
                index = np.argmax([t_ratio <= _r for _r in ratios[part] + [np.inf]])
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
            # TODO: rule out splits that have length < tol
            tree_t = Tree(
                n_labels=self.n_labels,
                splits=list(splits_t.keys()),
                lengths=gs.array(list(splits_t.values())),
            )
            return tree_t

        return geodesic_

    def _gtp_trees_with_common_support(self, splits_a, splits_b, tol=1e-8):
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
        tol: float
            Tolerance for the decision problem to avoid unambiguity.
            Defaults to 1e-08.

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

        trees_a = self._gtp_cut_tree_at_splits(total_a, cut_splits)
        trees_b = self._gtp_cut_tree_at_splits(total_b, cut_splits)
        supports = {
            part: self._gtp_trees_with_distinct_support(
                {s: splits_a[s] for s in trees_a[part]},
                {s: splits_b[s] for s in trees_b[part]},
                tol=tol,
            )
            for part in trees_a.keys()
            if trees_a[part] and trees_b[part]
        }
        common = common | easy_a | easy_b
        common_a = {s: splits_a[s] if s in sp_a else 0 for s in common}
        common_b = {s: splits_b[s] if s in sp_b else 0 for s in common}
        return common_a, common_b, supports

    def _gtp_cut_tree_at_splits(self, splits, cut_splits):
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

    def _gtp_trees_with_distinct_support(self, splits_a, splits_b, tol=10**-8):
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
        tol: float
            Tolerance for the decision problem to avoid unambiguity.
            Defaults to 1e-08.

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
            new_support_a, new_support_b = tuple(), tuple()
            for pair_a, pair_b in zip(old_support_a, old_support_b):
                pair_a_w = {s: weights_a[s] for s in pair_a}
                pair_b_w = {s: weights_b[s] for s in pair_b}
                value, c1, c2, d1, d2 = self._gtp_solve_extension_problem(
                    pair_a_w, pair_b_w
                )
                if value >= 1 - tol:
                    new_support_a += (pair_a,)
                    new_support_b += (pair_b,)
                else:
                    new_support_a += (c1, c2)
                    new_support_b += (d1, d2)
            if len(new_support_a) == len(old_support_a):
                return new_support_a, new_support_b
            else:
                old_support_a, old_support_b = new_support_a, new_support_b

    @staticmethod
    def _gtp_solve_extension_problem(sq_splits_a, sq_splits_b):
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
