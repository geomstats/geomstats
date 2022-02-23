""" Classes for trees and forests, splits and structures.

An instance of class Wald (Tree or Forest) is an element of the Wald Space or BHV space,
for example.

Lead author: Jonas Lueg
"""

import functools
import itertools as it

import geomstats.backend as gs


@functools.total_ordering
class Split:
    r""" A non-empty two-set partition of a set.

    We also allow splits to be a two-set partition of a smaller subset of {0,...,n-1},
    but the parameter ``n`` is still relevant for the hash function.

    The parameters ``part1`` and ``part2`` are assigned to the attributes ``self.part1``
    and ``self.part2`` sorted, and the one that contains the smallest minimal value is
    assigned to ``self.part1``, to obtain consistency.

    Parameters
    ----------
    n : int
        The number of labels, the set of labels will be :math:`\{0,\dots,n-1\}`.
    part1 : iterable
        The first part of the split, an iterable that is a subset of
        :math:`\{0,\dots,n-1\}`. It may be empty for technical reasons, but must have
        empty intersection with ``part2``.
    part2 : iterable
        The second part of the split, an iterable that is a subset of
        :math:`\{0,\dots,n-1\}`. It may be empty for technical reasons, but must have
        empty intersection with ``part1``.
    """

    def __init__(self, n, part1, part2):
        # sort both parts and convert them into tuples
        part1, part2 = tuple(sorted(list(part1))), tuple(sorted(list(part2)))
        if set(part1) & set(part2):
            raise ValueError(
                f"A split consists of disjoint sets, those are not: {part1}, {part2}.")
        # the next if-clauses make sure that we store the parts in a unique fixed way
        if part1 and part2:
            self._part1 = part1 if part1[0] < part2[0] else part2
            self._part2 = part2 if part1[0] < part2[0] else part1
        elif not part1:
            self._part1 = part2
            self._part2 = ()
        elif not part2:
            self._part1 = part1
            self._part2 = ()
        else:
            self._part1 = ()
            self._part2 = ()
        # TODO: the parameter n could be dropped theoretically...
        self._n = n

    @property
    def part1(self):
        """The first tuple of the split."""
        return self._part1

    @property
    def part2(self):
        """The second tuple of the split."""
        return self._part2

    @property
    def n(self):
        r"""Number of labels, the set of labels is then :math:`\{0,\dots,n-1\}`."""
        return self._n

    def restr(self, subset: set):
        """The restriction of a split to a subset, returns a split."""
        return Split(n=self.n, part1=tuple(set(self.part1) & subset),
                     part2=tuple(set(self.part2) & subset))

    def contains(self, subset: set):
        """ Determines if a subset is contained in either part of a split.

        Parameters
        ----------
        subset : set
            The subset.

        Returns
        -------
        is_contained : bool
            A boolean that is true if the subset is contained in ``self.part1`` or
            ``self.part2``.
        """
        return subset.issubset(set(self.part1)) or subset.issubset(set(self.part2))

    def separates(self, u, v):
        """ Determines whether the labels (or label sets) are separated by the split.

        Parameters
        ----------
        u : list of int, int
            Either an integer or a set of labels.
        v : list of int, int
            Either an integer or a set of labels.

        Returns
        -------
        is_separated : bool
            Determines whether u and v are separated by the split (i.e. if they are not
            in the same part).
        """
        if type(u) is type(v) is int:
            return (u in self.part1 and v in self.part2) or (
                    u in self.part2 and v in self.part1)
        b1 = set(u).issubset(set(self.part1)) and set(v).issubset(set(self.part2))
        b2 = set(v).issubset(set(self.part1)) and set(u).issubset(set(self.part2))
        return b1 or b2

    def point_to_split(self, other):
        """ Gives back the part of this split that is directed to the other split. """
        return self.part1 if set(self.part1) & set(other.part1) and set(
            self.part1) & set(other.part2) else self.part2

    def point_away_split(self, other):
        """ Gives back the part of this split that is directed away from other split."""
        return self.part2 if set(self.part1) & set(other.part1) and set(
            self.part1) & set(other.part2) else self.part1

    def compatible_with(self, other):
        """ Checks whether this split is compatible with another split.

        Two splits are compatible, if at least one intersection of the respective parts
        of the splits is empty.
        """
        p1, p2 = set(self.part1), set(self.part2)
        o1, o2 = set(other.part1), set(other.part2)
        return sum([bool(s) for s in [p1 & o1, p1 & o2, p2 & o1, p2 & o2]]) < 4

    def __eq__(self, other):
        """ The equal function. """
        return self.__hash__() == other.__hash__()

    def __lt__(self, other):
        """ Less than function (<). """
        return self.__hash__() < other.__hash__()

    def __hash__(self):
        """ The hash function. """
        return hash((self.n, self._part1, self._part2))

    def __str__(self):
        """ String representation of the split. """
        return str((self.part1, self.part2))

    def __repr__(self):
        return f"{self.part1}|{self.part2}"

    def __bool__(self):
        """ Returns False only if both parts are empty sets. """
        return bool(self.part1) and bool(self.part2)


class Structure:
    r""" A structure of a forest, that is the connected components and the edges.

    Parameters
    ----------
    n : int
        Number of labels, the set of labels is then :math:`\{0,\dots,n-1\}`.
    partition : tuple of tuple of int
        An iterable of iterables, namely a partition of the set :math:`\{0,\dots,n-1\}`,
        representing the connected components of the forest.
    split_sets : tuple of tuple of Split
        An iterable of iterables of splits, where each set of splits contains only
        splits of the respective connected component in the partition, so their order
        is related. The splits are then the edges of the connected components,
        respectively, and thus the union of all sets of splits is all edges of the
        forest.
    """

    def __init__(self, n, partition, split_sets):
        """ Initialize a structure with partition and sets of splits of components. """
        self._n = n
        # make some assertions about the given parameters.
        # same number of components in both partition and split_sets
        if len(split_sets) != len(partition):
            raise ValueError("Number of split sets is not equal to number of "
                             "components.")
        # all parts of partition give the whole set of labels.
        if set.union(*[set(part) for part in partition]) != set(range(n)):
            raise ValueError("partition is not a partition of the set (1,2,...,n).")
        # each split is a proper split of the label set of its component.
        for _part, _splits in zip(partition, split_sets):
            for _sp in _splits:
                if (set(_sp.part1) | set(_sp.part2)) != set(_part):
                    raise ValueError(f"The split {_sp} is not a split of component "
                                     f"{_part}.")

        # standardize the representation
        partition = [tuple(sorted(x)) for x in partition]
        # sort the partition itself, but also the split_sets.
        seq = [part[0] for part in partition]
        sort_key = sorted(range(len(seq)), key=seq.__getitem__)
        self._partition = tuple([partition[key] for key in sort_key])
        self._split_sets = tuple([tuple(sorted(split_sets[key])) for key in sort_key])
        self._leaf_paths = None
        self._separators = None
        self._support = None
        self._chart_gradient = None

    @property
    def n(self):
        r""" Number of labels, the set of labels is then :math:`\{0,\dots,n-1\}`. """
        return self._n

    @property
    def partition(self):
        """ The partition of the labels, each part is the label set of a component. """
        return self._partition

    @property
    def split_sets(self):
        """ Tuple of tuples of splits of the connected components of the forest. """
        return self._split_sets

    @property
    def leaf_paths(self):
        """ For each component, gives dictionary with splits on paths between labels.

        Returns
        -------
        leaf_paths : list of dict
            A list of dictionaries, each dictionary is for the respective connected
            component of the forest, and the items of each dictionary are for each pair
            of labels u, v in the respective component, a list of the splits on the
            unique path between the labels u and v.
        """
        if self._leaf_paths is None:
            self._leaf_paths = [
                {(u, v): [k for k, s in enumerate(splits) if s.separates(u, v)]
                 for u, v in it.combinations(self.partition[i], r=2)}
                for i, splits in enumerate(self.split_sets)]
        return self._leaf_paths

    @property
    def support(self):
        r""" For each split, gives a boolean matrix with entries indicating separation.

        Returns
        -------
        support : list of list
            A list (of same length as ``self.partition``) of lists, each of these lists
            is for the respective connected component and contains for each split in
            this connected component an :math:`n\times n` dimensional matrix, where the
            uv-th entry is ``True`` if the split separates the labels u and v, else
            ``False``.
        """
        if self._support is None:
            # for i-th component and k-th split in that component,
            # support[i][k][u, v] is True, if that split separates u and v.
            _support = [[gs.zeros((self.n, self.n), dtype=bool) for _ in splits] for
                        splits in self.split_sets]
            for i, d in enumerate(self.leaf_paths):
                for (u, v), split_list in d.items():
                    for k in split_list:
                        _support[i][k][u, v] = True
                        _support[i][k][v, u] = True
            self._support = self.unravel(_support)
        return self._support

    @property
    def sep(self):
        """ Gives list of indices indicating when new connected component starts.

        Returns
        -------
        separators : list of int
            An increasing list of numbers between 0 and ``len(self.x)``, starting with
            0, where each number indicates that a new connected component starts at that
            index. Useful for unraveling the vector ``self.x`` into lists of lists.
        """
        if self._separators is None:
            lengths = [len(splits) for splits in self.split_sets]
            self._separators = [0] + [sum(lengths[0:j + 1]) for j in
                                      range(len(lengths))]
        return self._separators

    def chart(self, x):
        """ Computes the chart of a grove with structure `self` at coordinate `x`.

        Parameters
        ----------
        x : array-like, shape=[n_edges]
            Takes a vector of length 'number of total splits' of the structure.

        Returns
        -------
        corr : array-like, shape=[n, n]
            Returns the corresponding correlation matrix.
        """
        _w = self.ravel(x=x)
        _corr = [[0 for _ in range(self.n)] for _ in range(self.n)]
        for i, d in enumerate(self.leaf_paths):
            for (u, v), split_indices in d.items():
                if len(split_indices):
                    _corr[u][v] = gs.prod(gs.array([1 - _w[i][k]
                                                    for k in split_indices]))
                else:
                    _corr[u][v] = 1
                _corr[v][u] = _corr[u][v]
        # TODO use gs.fill_diagonal here, but it is not implemented yet
        for u in range(self.n):
            _corr[u][u] = 1
        return gs.array(_corr, dtype=gs.float32)

    @property
    def chart_gradient(self):
        """ Computes the gradient of the chart of a grove with structure `st`.

        Returns
        -------
        chart_gradient : callable
            A map that takes as input a vector of length 'number of total splits',
            and returns a list of the partial derivatives of the map ``self.chart``.
        """
        if self._chart_gradient is None:
            def _chart_gradient(x):
                """ Input is a flat vector or list x (Nye parametrization). """
                coord_list = [[y if i != k else 0 for i, y in enumerate(x)] for k in
                              range(len(x))]
                _corr_gradient = [self.support[k] * -self.chart(xk)
                                  for k, xk in enumerate(coord_list)]
                return _corr_gradient

            self._chart_gradient = _chart_gradient
        return self._chart_gradient

    def where(self, s):
        """ Gives the index (unraveled) of the split s in the structure. """
        return int(gs.argmin([o != s for o in self.unravel(self.split_sets)]))

    def ravel(self, x):
        """ Transforms an iterable of length 'number of splits' into a list of lists."""
        return [x[i:j] for i, j in zip(self.sep[:-1], self.sep[1:])]

    @staticmethod
    def unravel(x):
        """ Concatenates all lists in a list (or other iterables). """
        return [y for z in x for y in z]

    def __str__(self):
        """ Prints the structure as a string representation. """
        return str(self.split_sets)

    def __repr__(self):
        """ Fancy string representation. """
        return str(self)

    def __eq__(self, other):
        """ Determines if two structures are equal. """
        # if partitions are not equal, then structures are not equal.
        equal_n = self.n == other.n
        equal_partition = self.partition == other.partition
        equal_split_sets = self.split_sets == other.split_sets
        return equal_n and equal_partition and equal_split_sets

    def __le__(self, other):
        """ Less than or equal function (<=).

        This method determines whether self < other with respect to the partial ordering
        introduced by [future publication].

        Parameters
        ----------
        other : Structure
            The structure to which self is compared to.
        """
        class MyCheckException(Exception):
            pass
        xs = [set(x) for x in self.partition]
        ys = [set(y) for y in other.partition]
        # ----- check out condition 1. of the partial ordering -----
        try:
            x_to_y = {i: [j for j, y in enumerate(ys) if x.issubset(y)][0] for i, x in
                      enumerate(xs)}
        except IndexError:  # in this case, x is not a refinement of y.
            return False
        # ----- check out condition 2. of the partial ordering -----
        try:
            for i, splits in enumerate(self.split_sets):
                # check if the splits are contained in the other restricted splits of
                # the corresponding component.
                restr_other_splits = {sp_y.restr(subset=xs[i]) for sp_y in
                                      other.split_sets[x_to_y[i]]}
                if not set(splits).issubset(restr_other_splits):
                    raise MyCheckException()
        except MyCheckException:
            return False
        # ----- check out condition 3. of the partial ordering -----
        try:
            for j in range(len(ys)):
                xs_in_y = [x for i, x in enumerate(xs) if x_to_y[i] == j]
                for x1, x2 in it.combinations(xs_in_y, r=2):
                    sep_sp = [sp for sp in other.split_sets[j] if sp.separates(x1, x2)]
                    if len(sep_sp) == 0:
                        raise MyCheckException()
        except MyCheckException:
            return False

        # ----- all conditions are satisfied -> it is indeed a partial ordering! -----
        return True

    def __gt__(self, other):
        """ Strictly greater than (>). """
        return other < self

    def __ge__(self, other):
        """ Greater than or equal function (>=). """
        return other <= self

    def __lt__(self, other):
        """ Less than function (<). """
        return self <= other and self != other

    def __hash__(self):
        """ Computes the hash of the structure. """
        return hash(str(self))


class Wald:
    r""" A class for phylogenetic forests, elements of the Wald space.

    Parameters
    ----------
    n : int
        Number of labels, the set of labels is then :math:`\{0,\dots,n-1\}`.
    st : Structure
        The structure of the forest.
    x : array-like
        The edge weights, array of floats between 0 and 1, with m entries, where m is
        the total number of splits/edges in the structure ``st``.
    """

    def __init__(self, n, st: Structure, x):
        self._n: int = n
        self._st: Structure = st
        self._x = x
        self._corr = None

    @property
    def st(self) -> Structure:
        """ The structure of the wald. """
        return self._st

    @property
    def x(self) -> gs.array:
        """ The (flat) vector containing the edge weights in Nye notation. """
        return self._x

    @property
    def n(self):
        """ The number of labels in the wald. """
        return self._n

    @property
    def corr(self):
        """ The correlation matrix representation of the forest. """
        if self._corr is None:
            self._corr = self.st.chart(self.x)
        return self._corr

    def __hash__(self):
        """ Computes the hash of the forest. """
        return hash((self.st, tuple(self.x)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str((self.st, tuple(self.x)))

    def __repr__(self):
        return str((self.st, tuple(self.x)))
