r"""Helper classes for tree spaces.

Class ``Split``.
Essentially, a ``Split`` is a two-set partition of a subset of :math:`\{0,\dots,n-1\}`.
This class is designed such that one part of both parts of the partition can be empty.
Splits are corresponding uniquely to edges in a phylogenetic forest, where, if one cuts
the edge in the forest, the resulting two-set partition of the labels of the respective
component of the forest is the corresponding split.


Lead author: Jonas Lueg
"""

import functools
import itertools

import geomstats.backend as gs


@functools.total_ordering
class Split:
    r"""Class for two-set partitions of sets.

    Two-set partitions of a smaller subset of :math:`\{0,...,n-1\}` are also allowed,
    where :math:`n` is a positive integer, which is not passed as an argument as it is
    nowhere needed.

    The parameters ``part1`` and ``part2`` are assigned to the attributes ``self.part1``
    and ``self.part2`` in a unique sorted way: the one that contains the smallest
    minimal value is assigned to ``self.part1`` for consistency.

    Parameters
    ----------
    part1 : iterable
        The first part of the split, an iterable that is a subset of
        :math:`\{0,\dots,n-1\}`. It may be empty, but must have empty intersection with
        ``part2``.
    part2 : iterable
        The second part of the split, an iterable that is a subset of
        :math:`\{0,\dots,n-1\}`. It may be empty, but must have empty intersection with
        ``part1``.
    """

    def __init__(self, part1, part2):
        part1, part2 = set(sorted(list(part1))), set(sorted(list(part2)))
        if part1 & part2:
            raise ValueError(
                f"A split consists of disjoint sets, those are not: {part1}, {part2}."
            )
        if part1 and part2:
            self.part1, self.part2 = (
                (part1, part2) if min(part1) < min(part2) else (part2, part1)
            )
        else:
            self.part1 = part1 or part2
            self.part2 = set()

    def __bool__(self):
        """Return True if and only if both parts are non-empty.

        We use the boolean representation to indicate whether both parts of a split are
        non-empty.

        Returns
        -------
        boolean_of_split : bool
            Returns the boolean representation of a split.
        """
        return bool(self.part1) and bool(self.part2)

    def __eq__(self, other):
        """Check for equal hashes of the two splits.

        Parameters
        ----------
        other : Split
            The other split.

        Returns
        -------
        is_equal : bool
            Return ``True`` if the splits are equal, else ``False``.
        """
        return hash(self) == hash(other)

    def __hash__(self):
        """Compute the hash of a split.

        Note that this hash simply uses the hash function for tuples.

        Returns
        -------
        hash_of_split : int
            Return the hash of the split.
        """
        return hash((tuple(self.part1), tuple(self.part2)))

    def __lt__(self, other):
        """Check if the hash of this split is less than the hash of the other split.

        Note that this partial ordering does not have a mathematical background, this is
        introduced in order to have a unique ordering for each set of splits at hand.

        Parameters
        ----------
        other : Split
            The other split.

        Returns
        -------
        is_strictly_less_than : bool
            Return ``True`` if hash is less than hash of other, else ``False``.
        """
        return hash(self) < hash(other)

    def __repr__(self):
        """Return the string representation of the split.

        This string representation requires that one can retrieve all necessary
        information from the string.

        Returns
        -------
        string_of_split : str
            Return the string representation of the split.
        """
        return str((self.part1, self.part2))

    def __str__(self):
        """Return the fancy printable string representation of the split.

        This string representation does NOT require that one can retrieve all necessary
        information from the string, but this string representation is required to be
        readable by a human.

        Returns
        -------
        string_of_split : str
            Return the fancy readable string representation of the split.
        """
        return f"{self.part1}|{self.part2}"

    def is_compatible(self, other):
        """Check whether this split is compatible with another split.

        Two splits are compatible, if at least one intersection of the respective parts
        of the splits is empty.

        Parameters
        ----------
        other : Split
            The other split.

        Returns
        -------
        is_compatible_with : bool
            Return ``True`` if the splits are compatible, else ``False``.
        """
        p1, p2 = self.part1, self.part2
        o1, o2 = other.part1, other.part2
        return sum([bool(s) for s in [p1 & o1, p1 & o2, p2 & o1, p2 & o2]]) < 4

    def get_part_away_from(self, other):
        """Return the part of this split that is directed away from other split.

        Parameters
        ----------
        other : Split
            The other split.

        Returns
        -------
        part_that_does_not_point : iterable
            Return the part of the split ``self`` that does not point toward
            ``other``. See ``self.get_part_towards`` for further explanation.
        """
        if other.part_contains(self.part1):
            return self.part1
        return self.part2

    def get_part_towards(self, other):
        """Return the part of this split that is directed toward the other split.

        Each split contains part1 and part2, the parts that the corresponding edge in
        the graph splits the set of labels into. Thus, one can think of the split as an
        edge, where part1 points in the direction of the part of the tree where the
        labels of part1 are contained, and part2 points in the other direction.
        So, part1 points in the direction of ``other``, if it corresponds to an
        edge that is contained in the tree that part1 points to, else part2 points in
        the direction of ``split``.

        Parameters
        ----------
        other : Split
            The other split.

        Returns
        -------
        part_towards : iterable
            Return the part of the split ``self`` that points toward ``other_split``.
        """
        if other.part_contains(self.part1):
            return self.part2
        return self.part1

    def part_contains(self, subset: set):
        """Determine if a subset is contained in either part of a split.

        Parameters
        ----------
        subset : set
            The subset containing labels.

        Returns
        -------
        is_contained : bool
            A boolean that is true if the subset is contained in ``self.part1`` or
            ``self.part2``.
        """
        return subset.issubset(self.part1) or subset.issubset(self.part2)

    def restrict_to(self, subset: set):
        r"""Return the restriction of a split to a subset.

        Parameters
        ----------
        subset : set
            The subset that the split is restricted to.

        Returns
        -------
        restr_split : Split
            The restricted split, if the split is :math:`A\vert B`, then the split
            restricted to the subset :math:`C` is :math:`A\cap C\vert B\cap C`.
        """
        return Split(
            part1=self.part1 & subset,
            part2=self.part2 & subset,
        )

    def separates(self, u, v):
        """Determine whether the labels (or label sets) are separated by the split.

        Parameters
        ----------
        u : list of int, int
            Either an integer or a set of labels.
        v : list of int, int
            Either an integer or a set of labels.

        Returns
        -------
        is_separated : bool
            A boolean determining whether u and v are separated by the split (i.e. if
             they are not in the same part).
        """
        u = {u} if isinstance(u, int) else set(u)
        v = {v} if isinstance(v, int) else set(v)
        b1 = u.issubset(self.part1) and v.issubset(self.part2)
        b2 = v.issubset(self.part1) and u.issubset(self.part2)
        return b1 or b2


class Topology:
    r"""The topology of a forest, using a split-based graph-structure representation.

    Parameters
    ----------
    n_labels : int
        Number of labels, the set of labels is then :math:`\{0,\dots,n-1\}`.
    partition : tuple
        A tuple of tuples that is a partition of the set :math:`\{0,\dots,n-1\}`,
        representing the label sets of each connected component of the forest topology.
    split_sets : tuple
        A tuple of tuples containing splits, where each set of splits contains only
        splits of the respective label set in the partition, so their order
        is related. The splits are the edges of the connected components of the forest,
        respectively, and thus the union of all sets of splits yields all edges of the
        forest topology.

    Attributes
    ----------
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

    def __init__(self, n_labels, partition, split_sets):
        """Return a topology with partition and sets of splits of components."""
        self._check_init(n_labels, partition, split_sets)

        self.n_labels = n_labels
        partition = [tuple(sorted(x)) for x in partition]
        seq = [part[0] for part in partition]
        sort_key = sorted(range(len(seq)), key=seq.__getitem__)
        self.partition = tuple([partition[key] for key in sort_key])
        self.split_sets = tuple([tuple(sorted(split_sets[key])) for key in sort_key])

        self.where = {s: i for i, s in enumerate(self.flatten(self.split_sets))}

        lengths = [len(splits) for splits in self.split_sets]
        self.sep = [0] + [sum(lengths[0:j]) for j in range(1, len(lengths) + 1)]

        self.paths = [
            {
                (u, v): [s for s in splits if s.separates(u, v)]
                for u, v in itertools.combinations(part, r=2)
            }
            for part, splits in zip(self.partition, self.split_sets)
        ]

        _support = [
            gs.zeros((self.n_labels, self.n_labels), dtype=bool)
            for _ in self.flatten(self.split_sets)
        ]
        for path_dict in self.paths:
            for (u, v), path in path_dict.items():
                for split in path:
                    _support[self.where[split]][u][v] = True
                    _support[self.where[split]][v][u] = True
        self.support = [gs.array(m) for m in self.flatten(_support)]
        self._chart_gradient = None

    def _check_init(self, n_labels, partition, split_sets):
        if len(split_sets) != len(partition):
            raise ValueError(
                "Number of split sets is not equal to number of " "components."
            )
        if set.union(*[set(part) for part in partition]) != set(range(n_labels)):
            raise ValueError("The partition is not a partition of the set (0,...,n-1).")
        for _part, _splits in zip(partition, split_sets):
            for _sp in _splits:
                if (_sp.part1 | _sp.part2) != set(_part):
                    raise ValueError(
                        f"The split {_sp} is not a split of component {_part}."
                    )

    def __eq__(self, other):
        """Check if ``self`` is equal to ``other``.

        Parameters
        ----------
        other : Topology
            The other topology.

        Returns
        -------
        is_equal : bool
            Return ``True`` if the topologies are equal, else ``False``.
        """
        equal_n = self.n_labels == other.n_labels
        equal_partition = self.partition == other.partition
        equal_split_sets = self.split_sets == other.split_sets
        return equal_n and equal_partition and equal_split_sets

    def __ge__(self, other):
        """Check if ``self`` is greater than or equal to ``other``.

        Parameters
        ----------
        other : Topology
            The other topology.

        Returns
        -------
        is_greater_than_or_equal : bool
            Return ``True`` if ``self`` is greater or equal to ``other``, else
            ``False``.
        """
        return other <= self

    def __gt__(self, other):
        """Check if ``self`` is strictly greater than ``other``.

        Parameters
        ----------
        other : Topology
            The other topology.

        Returns
        -------
        is_greater_than : bool
            Return ``True`` if this topology is greater than the other, else ``False``.
        """
        return other < self

    def __hash__(self):
        """Compute the hash of a topology.

        Note that this hash simply uses the hash function for tuples.

        Returns
        -------
        hash_of_topology : int
            Return the hash of the topology.
        """
        return hash((self.n_labels, self.partition, self.split_sets))

    def __le__(self, other):
        """Check if ``self`` is less than or equal to ``other``.

        This partial ordering is the one defined in [1] and to show if self <= other is
        True, three things must be satisfied.
        (i)     ``self.partition`` must be a refinement of ``other.partition`` in the
                sense of partitions.
        (ii)    The splits of each component in ``self`` must be contained in the
                set of splits of ``other`` restricted to the component of ``self``.
        (iii)   Whenever two components of ``self`` are contained in a component of
                ``other``, there needs to exist a split in ``other`` separating those
                two components.
        If one of those three conditions are not fulfilled, this method returns False.

        Parameters
        ----------
        other : Topology
            The structure to which self is compared to.

        Returns
        -------
        is_less_than_or_equal : bool
            Return ``True`` if (i), (ii) and (iii) are satisfied, else ``False``.
        """

        class NotPartialOrder(Exception):
            """Raise an exception when less equal is not true."""

        x_parts = [set(x) for x in self.partition]
        y_parts = [set(y) for y in other.partition]
        # (i)
        try:
            cover = {
                i: [j for j, y in enumerate(y_parts) if x.issubset(y)][0]
                for i, x in enumerate(x_parts)
            }
        except IndexError:
            return False
        # (ii)
        try:
            for (i, j), x in zip(cover.items(), x_parts):
                y_splits_restricted = {
                    split_y.restrict_to(subset=x) for split_y in other.split_sets[j]
                }
                if not set(self.split_sets[i]).issubset(y_splits_restricted):
                    raise NotPartialOrder()
        except NotPartialOrder:
            return False
        # (iii)
        try:
            for j in range(len(y_parts)):
                xs_in_y = [x for i, x in enumerate(x_parts) if cover[i] == j]
                for x1, x2 in itertools.combinations(xs_in_y, r=2):
                    sep_sp = [sp for sp in other.split_sets[j] if sp.separates(x1, x2)]
                    if not sep_sp:
                        raise NotPartialOrder()
        except NotPartialOrder:
            return False
        return True

    def __lt__(self, other):
        """Check if ``self`` is less than ``other``.

        Parameters
        ----------
        other : Topology
            The other topology.

        Returns
        -------
        is_less_than : bool
            Return ``True`` if ``self`` less than ``other``, else ``False``.
        """
        return self <= other and self != other

    def __repr__(self):
        """Return the string representation of the topology.

        This string representation requires that one can retrieve all necessary
        information from the string.

        Returns
        -------
        string_of_topology : str
            Return the string representation of the topology.
        """
        return str((self.n_labels, self.partition, self.split_sets))

    def __str__(self):
        """Return the fancy printable string representation of the topology.

        This string representation does NOT require that one can retrieve all necessary
        information from the string, but this string representation is required to be
        readable by a human.

        Returns
        -------
        string_of_topology : str
            Return the fancy readable string representation of the topology.
        """
        comps = [", ".join(str(sp) for sp in splits) for splits in self.split_sets]
        return "(" + "; ".join(comps) + ")"

    def corr(self, x):
        """Compute the correlation matrix of the topology with edge weights ``weights``.

        Parameters
        ----------
        x : array-like
            Takes a vector of length 'number of total splits' of the structure.

        Returns
        -------
        corr : array-like, shape=[n, n]
            Returns the corresponding correlation matrix.
        """
        corr = gs.zeros((self.n_labels, self.n_labels))
        for path_dict in self.paths:
            for (u, v), path in path_dict.items():
                corr[u][v] = gs.prod([1 - x[self.where[split]] for split in path])
                corr[v][u] = corr[u][v]

        corr = gs.array(corr)
        return corr + gs.eye(corr.shape[0])

    def corr_gradient(self, x):
        """Compute the gradient of the correlation matrix, differentiated by weights.

        Parameters
        ----------
        x : array-like
            The vector weights at which the gradient is computed.

        Returns
        -------
        gradient : array-like, shape=[n_splits, n, n]
            The gradient of the correlation matrix, differentiated by weights.
        """
        x_list = [[y if i != k else 0 for i, y in enumerate(x)] for k in range(len(x))]
        gradient = gs.array(
            [-supp * self.corr(x) for supp, x in zip(self.support, x_list)]
        )
        return gradient

    def unflatten(self, x):
        """Transform list into list of lists according to separators, ``self.sep``.

        The separators are a list of integers, increasing. Then, all elements between to
        indices in separators will be put into a list, and together, all lists give a
        nested list.

        Parameters
        ----------
        x : iterable
            The flat list that will be nested.

        Returns
        -------
        x_nested : list[list]
            The nested list of lists.
        """
        return [x[i:j] for i, j in zip(self.sep[:-1], self.sep[1:])]

    @staticmethod
    def flatten(x):
        """Flatten a list of lists into a single list by concatenation.

        Parameters
        ----------
        x : nested list
            The nested list to flatten.

        Returns
        -------
        x_flat : list, tuple
            The flatted list.
        """
        return [y for z in x for y in z]
