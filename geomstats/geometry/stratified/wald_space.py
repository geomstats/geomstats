r"""Classes for the Wald Space and elements therein of class Wald and helper classes.

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
.. [Garba21] Garba, M. K., T. M. W. Nye, J. Lueg and S. F. Huckemann.
    "Information geometry for phylogenetic trees"
    Journal of Mathematical Biology, 82(3):19, February 2021a.
    https://doi.org/10.1007/s00285-021-01553-x.
.. [Lueg21] Lueg, J., M. K. Garba, T. M. W. Nye, S. F. Huckemann.
    "Wald Space for Phylogenetic Trees."
    Geometric Science of Information, Lecture Notes in Computer Science,
    pages 710–717, Cham, 2021.
    https://doi.org/10.1007/978-3-030-80209-7_76.
"""

import itertools

import geomstats.backend as gs
import geomstats.geometry.spd_matrices as spd
from geomstats.geometry.stratified.point_set import Point, PointSet, _vectorize_point
from geomstats.geometry.stratified.tree import Split, Topology


class Wald(Point):
    r"""A class for wälder, that are phylogenetic forests, elements of the Wald Space.

    Parameters
    ----------
    topology : Topology
        The structure of the forest.
    weights : array-like
        The edge weights, array of floats between 0 and 1, with m entries, where m is
        the total number of splits/edges in the structure ``top``.
    """

    def __init__(self, topology, weights):
        super().__init__()
        self.topology = topology
        self.weights = weights
        self.corr = self.topology.corr(weights)

    @property
    def n_labels(self):
        """Get number of labels."""
        return self.topology.n_labels

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
        return hash((self.topology, tuple(self.weights)))

    def __repr__(self):
        """Return the string representation of the wald.

        This string representation requires that one can retrieve all necessary
        information from the string.

        Returns
        -------
        string_of_wald : str
            Return the string representation of the wald.
        """
        return repr((self.topology, tuple(self.weights)))

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
        return f"({str(self.topology)};{str(self.weights)})"

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
                for u, v in itertools.combinations(labels, 2)
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
    def _generate_partition(n_labels, p_new):
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
        for u in range(1, n_labels):
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
    def generate_wald(n_labels, p_keep, p_new, btol=10**-8, check=True):
        """Generate a random instance of class ``Wald``.

        Parameters
        ----------
        n_labels : int
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
        partition = Wald._generate_partition(n_labels=n_labels, p_new=p_new)
        split_sets = [Wald._generate_splits(labels=_part) for _part in partition]
        split_sets = [
            Wald._delete_splits(splits=splits, labels=part, p_keep=p_keep, check=check)
            for part, splits in zip(partition, split_sets)
        ]
        top = Topology(n_labels=n_labels, partition=partition, split_sets=split_sets)
        x = gs.random.uniform(size=(len(top.flatten(split_sets)),), low=0, high=1)
        x = gs.minimum(gs.maximum(btol, x), 1 - btol)
        return Wald(topology=top, weights=x)


class WaldSpace(PointSet):
    """Class for the Wald space, a metric space for phylogenetic forests.

    Parameters
    ----------
    n_labels : int
        Integer determining the number of labels in the forests, and thus the shape of
        the correlation matrices: n_labels x n_labels.

    Attributes
    ----------
    ambient :
        The ambient space, the positive definite n_labels x n_labels matrices that the
        WaldSpace is embedded into.
    """

    def __init__(self, n_labels):
        super().__init__(equip=False)
        self.n_labels = n_labels
        self.ambient = spd.SPDMatrices(n=self.n_labels)

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
        p_new = p_tree ** (1 / (self.n_labels - 1))
        sample = [
            Wald.generate_wald(self.n_labels, p_keep, p_new, btol, check=True)
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
