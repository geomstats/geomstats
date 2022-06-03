r"""Classes for the Wald Space and elements therein of class Wald and helper classes.

Class ``Split``.
Essentially, a ``Split`` is a two-set partition of a subset of :math:`\{0,\dots,n-1\}`.
This class is designed such that one part of both parts of the partition can be empty.
Splits are corresponding uniquely to edges in a phylogenetic forest, where, if one cuts
the edge in the forest, the resulting two-set partition of the labels of the respective
component of the forest is the corresponding split.

Class ``Structure``.
A structure is a partition into non-empty sets of the set :math:`\{0,\dots,n-1\}`,
together with a set of splits for each element of the partition, where every split is a
two-set partition of the respective element.
A structure basically describes a phylogenetic forest, where each set of splits gives
the structure of the tree with the labels of the corresponding element of the partition.

Class ``Wald``.
A wald is essentially a phylogenetic forest with weights between zero and one on the
edges. The forest structure is stored as a ``Structure`` and the edge weights are an
array of length that is equal to the total number of splits in the structure. These
elements are the points in Wald space and other phylogenetic forest spaces, like BHV
space, although the partition is just the whole set of labels in this case.

Class ``WaldSpace``.
A topological space. Points in Wald space are instances of the class :class:`Wald`:
phylogenetic forests with edge weights between 0 and 1.
In particular, Wald space is a stratified space, each stratum is called grove.
The highest dimensional groves correspond to fully resolved or binary trees.
The topology is obtained from embedding wälder into the ambient space of strictly
positive :math:`n\times n` symmetric matrices, implemented in the
class :class:`spd.SPDMatrices`.


Lead author: Jonas Lueg


References
----------
[Garba21]_  Garba, M. K., T. M. W. Nye, J. Lueg and S. F. Huckemann.
            "Information geometry for phylogenetic trees"
            Journal of Mathematical Biology, 82(3):19, February 2021a.
            https://doi.org/10.1007/s00285-021-01553-x.
[Lueg21]_   Lueg, J., M. K. Garba, T. M. W. Nye, S. F. Huckemann.
            "Wald Space for Phylogenetic Trees."
            Geometric Science of Information, Lecture Notes in Computer Science,
            pages 710–717, Cham, 2021.
            https://doi.org/10.1007/978-3-030-80209-7_76.
"""

import functools
import itertools as it

import geomstats.backend as gs
import geomstats.geometry.spd_matrices as spd
from geomstats.geometry.stratified.point_set import Point, PointSet, _vectorize_point


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
    n : int
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

    def __init__(self, n, partition, split_sets):
        """Return a topology with partition and sets of splits of components."""
        if len(split_sets) != len(partition):
            raise ValueError(
                "Number of split sets is not equal to number of " "components."
            )
        if set.union(*[set(part) for part in partition]) != set(range(n)):
            raise ValueError("The partition is not a partition of the set (0,...,n-1).")
        for _part, _splits in zip(partition, split_sets):
            for _sp in _splits:
                if (_sp.part1 | _sp.part2) != set(_part):
                    raise ValueError(
                        f"The split {_sp} is not a split of component {_part}."
                    )

        self.n = n
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
                for u, v in it.combinations(part, r=2)
            }
            for part, splits in zip(self.partition, self.split_sets)
        ]

        _support = [
            gs.zeros((self.n, self.n), dtype=bool)
            for _ in self.flatten(self.split_sets)
        ]
        for path_dict in self.paths:
            for (u, v), path in path_dict.items():
                for split in path:
                    _support[self.where[split]][u][v] = True
                    _support[self.where[split]][v][u] = True
        self.support = [gs.array(m) for m in self.flatten(_support)]
        self._chart_gradient = None

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
        equal_n = self.n == other.n
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
        return hash((self.n, self.partition, self.split_sets))

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
                for x1, x2 in it.combinations(xs_in_y, r=2):
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
        return str((self.n, self.partition, self.split_sets))

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
        corr = gs.zeros((self.n, self.n))
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


class Wald(Point):
    r"""A class for wälder, that are phylogenetic forests, elements of the Wald Space.

    Parameters
    ----------
    n : int
        Number of labels, the set of labels is then :math:`\{0,\dots,n-1\}`.
    topology : Topology
        The structure of the forest.
    weights : array-like
        The edge weights, array of floats between 0 and 1, with m entries, where m is
        the total number of splits/edges in the structure ``top``.
    """

    def __init__(self, n: int, topology: Topology, weights):
        # TODO: make `n` come from topology?
        # TODO: keep parameters naming
        super().__init__()
        self.n = n
        self.top = topology
        self.weights = weights
        self.corr = self.top.corr(weights)

    def __eq__(self, other):
        """Check for equal hashes of the two wälder.

        Parameters
        ----------
        other : Wald
            The other wald.

        Returns
        -------
        is_equal : bool
            Return ``True`` if the wälder are equal, else ``False``.
        """
        return hash(self) == hash(other)

    def __hash__(self):
        """Compute the hash of the wald.

        Note that this hash simply uses the hash function for tuples.

        Returns
        -------
        hash_of_wald : int
            Return the hash of the wald.
        """
        return hash((self.top, tuple(self.weights)))

    def __repr__(self):
        """Return the string representation of the wald.

        This string representation requires that one can retrieve all necessary
        information from the string.

        Returns
        -------
        string_of_wald : str
            Return the string representation of the wald.
        """
        return repr((self.top, tuple(self.weights)))

    def __str__(self):
        """Return the fancy printable string representation of the wald.

        This string representation does NOT require that one can retrieve all necessary
        information from the string, but this string representation is required to be
        readable by a human.

        Returns
        -------
        string_of_wald : str
            Return the fancy readable string representation of the wald.
        """
        return f"({str(self.top)};{str(self.weights)})"

    def to_array(self):
        """Turn the wald into a numpy array, namely its correlation matrix.

        Returns
        -------
        array_of_wald : array-like, shape=[n, n]
            The correlation matrix corresponding to the wald.
        """
        return self.corr

    @staticmethod
    def _check_if_separated(labels, splits):
        """Check for each pair of labels if exists split that separates them.

        Parameters
        ----------
        labels : list[int]
            A list of integers, the set of labels that we generate splits for.
        splits : list[Split]
            A list of splits of the set of labels.

        Returns
        -------
        are_separated : bool
            True if the labels are pair-wise separated by a split else False.
        """
        return gs.all(
            [
                gs.any([sp.separates(u, v) for sp in splits])
                for u, v in it.combinations(labels, 2)
            ]
        )

    @staticmethod
    def _delete_splits(splits, labels, p_keep, check=True):
        """Delete splits randomly from a set of splits.

        We require the splits to satisfy the check for if all pair-wise labels are
        separated. In this way, before deleting a split, this condition is checked
        to make sure it is not violated.

        Parameters
        ----------
        splits : list[Split]
            A list of splits of the set of labels.
        labels : list[int]
            A list of integers, the set of labels that we generate splits for.
        p_keep : float
            A float between 0 and 1 determining the probability with which a split
            is kept and not deleted.
        check : bool
            If True, checks if splits still separate all labels. In this case, the split
            will not be deleted. If False, any split can be randomly deleted.

        Returns
        -------
        left_over_splits : list[Split]
            The list of splits that are not deleted.
        """
        if p_keep == 1:
            return splits
        for i in reversed(range(len(splits))):
            if gs.random.rand(1) > p_keep:
                splits_cp = splits.copy()
                splits_cp.pop(i)
                if not check:
                    splits = splits_cp
                elif Wald._check_if_separated(splits=splits_cp, labels=labels):
                    splits = splits_cp
        return splits

    @staticmethod
    def _generate_partition(n, p_new):
        r"""Generate a random partition of :math:`\{0,\dots,n-1\}`.

        This algorithm works as follows: Start with a single set containing zero,
        then successively add the labels from 1 to n-1 to the partition in the
        following manner: for each label u, with probability `probability`, add the
        label u to a random existing set of the partition, else add a new singleton
        set {u} to the partition (i.e. with probability 1 - `probability`).

        Parameters
        ----------
        p_new : float
            A float between 0 and 1, the probability that no new component is added,
            and 1 - probability that a new component is added.

        Returns
        -------
        partition : list[list[int]]
            A partition of the set :math:`\{0,\dots,n-1\}` into non-empty sets.
        """
        _partition = [[0]]
        for u in range(1, n):
            if gs.random.rand(1) < p_new:
                index = int(gs.random.randint(0, len(_partition), (1,)))
                _partition[index].append(u)
            else:
                _partition.append([u])
        return _partition

    @staticmethod
    def _generate_splits(labels):
        """Generate random maximal set of compatible splits of set ``labels``.

        This method works inductively on the number of elements in labels.
        Start with a split of two randomly chosen labels. Then, successively choose
        a label from the labels and add this as a leaf with a split to the existing
        tree by attaching it to a random split, thereby dividing this split into two
        splits and one has to update all the other splits accordingly.

        Parameters
        ----------
        labels : list[int]
            A list of integers, the set of labels that we generate splits for.

        Returns
        -------
        splits : list[Split]
            A list of splits of the set of labels, maximal number of splits.
        """
        if len(labels) <= 1:
            return []
        unused_labels = labels.copy()
        random_index = int(gs.random.randint(0, len(unused_labels), (1,)))
        u = unused_labels.pop(random_index)
        random_index = int(gs.random.randint(0, len(unused_labels), (1,)))
        v = unused_labels.pop(random_index)
        used_labels = [u, v]
        splits = [Split(part1={u}, part2={v})]
        while unused_labels:
            random_index = int(gs.random.randint(0, len(unused_labels), (1,)))
            u = unused_labels.pop(random_index)
            updated_splits = [Split(part1={u}, part2=used_labels)]
            random_index = int(gs.random.randint(0, len(splits), (1,)))
            divided_split = splits.pop(random_index)
            updated_splits.append(
                Split(part1=divided_split.part1 | {u}, part2=divided_split.part2)
            )
            updated_splits.append(
                Split(part1=divided_split.part1, part2=divided_split.part2 | {u})
            )
            for split in splits:
                updated_splits.append(
                    Split(
                        part1=split.get_part_away_from(divided_split),
                        part2=split.get_part_towards(divided_split) | {u},
                    )
                )
            used_labels.append(u)
            splits = updated_splits
        return splits

    @staticmethod
    def generate_wald(n, p_keep, p_new, btol=10**-8, check=True):
        """Generate a random instance of class ``Wald``.

        Parameters
        ----------
        n : int
            The number of labels the wald is generated with respect to.
        p_keep : float
            The probability will be inserted into the generation of a partition as
            well as for the generation of a split set for the topology of the wald.
        p_new : float
            A float between 0 and 1, the probability that no new component is added,
            and probability of 1 - p_new_ that a new component is added.
        btol: float
            Tolerance for the boundary of the coordinates in each grove. Defaults to
            1e-08.
        check : bool
            If True, checks if splits still separate all labels. In this case, the split
            will not be deleted. If False, any split can be randomly deleted.

        Returns
        -------
        random_wald : Wald
            The randomly generated wald.
        """
        partition = Wald._generate_partition(n=n, p_new=p_new)
        split_sets = [Wald._generate_splits(labels=_part) for _part in partition]
        split_sets = [
            Wald._delete_splits(splits=splits, labels=part, p_keep=p_keep, check=check)
            for part, splits in zip(partition, split_sets)
        ]
        top = Topology(n=n, partition=partition, split_sets=split_sets)
        x = gs.random.uniform(size=(len(top.flatten(split_sets)),), low=0, high=1)
        x = gs.minimum(gs.maximum(btol, x), 1 - btol)
        return Wald(n=n, topology=top, weights=x)


class WaldSpace(PointSet):
    """Class for the Wald space, a metric space for phylogenetic forests.

    Parameters
    ----------
    n : int
        Integer determining the number of labels in the forests, and thus the shape of
        the correlation matrices: n x n.

    Attributes
    ----------
    ambient :
        The ambient space, the positive definite n x n matrices that the WaldSpace is
        embedded into.
    """

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.ambient = spd.SPDMatrices(n=self.n)

    @_vectorize_point((1, "point"))
    def belongs(self, point):
        """Check if a point `wald` belongs to Wald space.

        From FUTURE PUBLICATION we know that the corresponding matrix of a wald is
        strictly positive definite if and only if the labels are separated by at least
        one edge, which is exactly when the wald is an element of the Wald space.

        Parameters
        ----------
        point : Wald or list of Wald
            The point to be checked.

        Returns
        -------
        belongs : bool
            Boolean denoting if `point` belongs to Wald space.
        """
        is_spd = [
            self.ambient.belongs(single_point.to_array()) for single_point in point
        ]
        is_between_0_1 = [
            gs.all(w.weights > 0) and gs.all(w.weights < 1) for w in point
        ]
        results = [is1 and is2 for is1, is2 in zip(is_spd, is_between_0_1)]
        return results

    def random_point(self, n_samples=1, p_tree=0.9, p_keep=0.9, btol=1e-08):
        """Sample a random point in Wald space.

        Parameters
        ----------
        n_samples : int
            Number of samples. Defaults to 1.
        p_tree : float between 0 and 1
            The probability that the sampled point is a tree, and not a forest. If the
            probability is equal to 1, then the sampled point will be a tree.
            Defaults to 0.9.
        p_keep : float between 0 and 1
            The probability that a sampled edge is kept and not deleted randomly.
            To be precise, it is not exactly the probability, as some edges cannot be
            deleted since the requirement that two labels are separated by a split might
            be violated otherwise.
            Defaults to 0.9
        btol: float
            Tolerance for the boundary of the coordinates in each grove. Defaults to
            1e-08.

        Returns
        -------
        samples : Wald or list of Wald, shape=[n_samples]
            Points sampled in Wald space.
        """
        p_new = p_tree ** (1 / (self.n - 1))
        sample = [
            Wald.generate_wald(self.n, p_keep, p_new, btol, check=True)
            for _ in range(n_samples)
        ]
        return sample

    @_vectorize_point((1, "point"))
    def set_to_array(self, points):
        """Convert a set of points into an array.

        Parameters
        ----------
        points : list of Wald, shape=[...]
            Number of samples of wälder to turn into an array.

        Returns
        -------
        points_array : array-like, shape=[...]
            Array of the wälder that are turned into arrays.
        """
        results = gs.array([wald.to_array() for wald in points])
        return results
