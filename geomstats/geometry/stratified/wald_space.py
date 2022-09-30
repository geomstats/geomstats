r"""Classes for the Wald Space and elements therein of class Wald and helper classes.

Class ``Topology``.
A structure is a partition into non-empty sets of the set :math:`\{0,\dots,n-1\}`,
together with a set of splits for each element of the partition, where every split is a
two-set partition of the respective element.
A structure basically describes a phylogenetic forest, where each set of splits gives
the structure of the tree with the labels of the corresponding element of the partition.

Class ``Wald``.
A wald is essentially a phylogenetic forest with weights between zero and one on the
edges. The forest structure is stored as a ``Topology`` and the edge weights are an
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

import scipy

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricEuclidean
from geomstats.geometry.stratified.point_set import (
    Point,
    PointSet,
    PointSetMetric,
    _vectorize_point,
)
from geomstats.geometry.stratified.trees import BaseTopology as Topology
from geomstats.geometry.stratified.trees import Split, delete_splits, generate_splits


def make_splits(n_labels):
    """Generates all possible splits of a collection."""
    if n_labels <= 1:
        raise ValueError("`n_labels` must be greater than 1.")
    if n_labels == 2:
        yield Split(part1=[0], part2=[1])
    else:
        for split in make_splits(n_labels=n_labels - 1):
            yield Split(part1=split.part1, part2=split.part2.union((n_labels - 1,)))
            yield Split(part1=split.part1.union((n_labels - 1,)), part2=split.part2)
        yield Split(part1=list(range(n_labels - 1)), part2=[n_labels - 1])


def make_topologies(n_labels):
    """Generates all possible sets of compatible splits of a collection.

    This only works well for `len(n_labels) < 8`.
    """
    if n_labels <= 1:
        raise ValueError("The collection must have 2 elements or more.")
    if n_labels in [2, 3]:
        yield Topology(
            n_labels=n_labels,
            partition=(tuple(range(n_labels)),),
            split_sets=(list(make_splits(n_labels)),),
        )
    else:
        pendant_split = Split(part1=[n_labels - 1], part2=list(range(n_labels - 1)))
        for st in make_topologies(n_labels - 1):
            for s in st.split_sets[0]:
                new_split_set = [pendant_split]
                a, b = set(s.part1), set(s.part2)
                for t in st.split_sets[0]:
                    c, d = set(t.part1), set(t.part2)
                    if t != s:
                        # TODO: probably a bug here
                        if a.issubset(d) or b.issubset(d):
                            new_split_set.append(
                                Split(
                                    part1=t.part1, part2=t.part2.union((n_labels - 1,))
                                )
                            )
                        else:
                            new_split_set.append(
                                Split(
                                    part1=t.part2, part2=t.part1.union((n_labels - 1,))
                                )
                            )
                    else:
                        new_split_set.append(
                            Split(part1=s.part1, part2=s.part2.union((n_labels - 1,)))
                        )
                        new_split_set.append(
                            Split(part1=s.part2, part2=s.part1.union((n_labels - 1,)))
                        )
                yield Topology(
                    n_labels=n_labels,
                    partition=(tuple(range(n_labels)),),
                    split_sets=(new_split_set,),
                )


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


def generate_random_wald(n_labels, p_keep, p_new, btol=1e-8, check=True):
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
    partition = _generate_partition(n_labels=n_labels, p_new=p_new)
    split_sets = [generate_splits(labels=_part) for _part in partition]

    split_sets = [
        delete_splits(splits=splits, labels=part, p_keep=p_keep, check=check)
        for part, splits in zip(partition, split_sets)
    ]

    top = Topology(n_labels=n_labels, partition=partition, split_sets=split_sets)
    x = gs.random.uniform(size=(len(top.flatten(split_sets)),), low=0, high=1)
    x = gs.minimum(gs.maximum(btol, x), 1 - btol)
    return Wald(topology=top, weights=x)


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
        # TODO: need to be consistent with BHV space
        super().__init__()
        self.topology = topology
        self.weights = weights
        # TODO: do we need to compute it? specially for BHV space
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


class WaldSpace(PointSet):
    """Class for the Wald space, a metric space for phylogenetic forests.

    Parameters
    ----------
    n_labels : int
        Integer determining the number of labels in the forests, and thus the shape of
        the correlation matrices: n_labels x n_labels.

    Attributes
    ----------
    ambient_space : Manifold
        The ambient space, the positive definite n_labels x n_labels matrices that the
        WaldSpace is embedded into.
    """

    def __init__(self, n_labels, ambient_space=None):
        super().__init__()
        self.n_labels = n_labels

        if ambient_space is None:
            ambient_space = SPDMatrices(n=self.n_labels)
        self.ambient_space = ambient_space

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
            self.ambient_space.belongs(single_point.to_array())
            for single_point in point
        ]
        is_between_0_1 = [
            gs.all(w.weights > 0) and gs.all(w.weights < 1) for w in point
        ]
        results = [is1 and is2 for is1, is2 in zip(is_spd, is_between_0_1)]
        return results

    def random_point(self, n_samples=1, p_tree=0.9, p_keep=0.9, btol=1e-8):
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
        forests = [
            generate_random_wald(self.n_labels, p_keep, p_new, btol, check=True)
            for _ in range(n_samples)
        ]

        if n_samples == 1:
            return forests[0]

        return forests

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


class WaldSpaceMetric(PointSetMetric):
    # TODO: delete

    # geodesic algorithms

    # naive needs: s_proj, a_path

    # symmetric needs: s_proj, a_path_t

    # s_proj needs _proj_target_gradient (changes with ambient metric)
    # _proj_target_gradient needs s_chart_and_gradient
    # s_chart and s_chart_gradient available in ftools and do not depend in ambient metric

    # straightning-ext needs: a_log, a_exp, s_proj, starting path (e.g. naive)
    def __init__(self, space, projection_solver):
        super().__init__(space)
        self.projection_solver = projection_solver
        # TODO: geodesic algorithm?

    @property
    def stratum_metric(self):
        return self.space.ambient_space.metric

    def dist(self):
        pass

    def geodesic(self):
        pass

    def lift(self, point):
        """Lift a point to the ambient space.

        Returns
        -------
        ambient_point : array-like, shape=[..., n_labels, n_labels]
        """
        # TODO: is extend a better name?
        # TODO: here or in space? (probably in space, since not metric dep)
        return point.corr

    def projection(self, ambient_point, **kwargs):
        """Projects a point into Wald space."""
        return self.projection_solver.projection(self, ambient_point, **kwargs)


class BaseProjectionSolver:
    def __init__(self):
        self._map_ambient_metric_to_target_gradient = {
            SPDMetricEuclidean: self._euclidean_target_gradient,
        }

    def _euclidean_target_gradient(self, weights, topology, ambient_point, metric):
        # TODO: should this be based on lift and lift_grad?
        corr = topology.corr(weights)
        grad = topology.corr_gradient(weights)

        target = metric.stratum_metric.squared_dist(corr, ambient_point)
        target_grad = gs.array(
            [2 * gs.sum((corr - ambient_point) * grad_) for grad_ in grad]
        )

        return target, target_grad

    def _proj_target_gradient(self, metric, ambient_point, topology):
        metric_target_gradient = self._map_ambient_metric_to_target_gradient[
            type(metric.stratum_metric)
        ]

        return lambda x: metric_target_gradient(
            weights=x, topology=topology, ambient_point=ambient_point, metric=metric
        )


class LocalProjectionSolver(BaseProjectionSolver):
    def __init__(self, btol=1e-10, **kwargs):
        super().__init__()
        self._minimize = scipy.optimize.minimize

        self.btol = btol
        self.optimization_kwargs = dict(
            jac=True,
            method="L-BFGS-B",
            tol=None,
            options=dict(gtol=1e-5, ftol=2.22e-9),
        )
        self.optimization_kwargs.update(kwargs)

    def _get_bounds(self, n_splits):
        return [(self.btol, 1 - self.btol)] * n_splits

    def projection(self, metric, ambient_point, topology):
        if len(topology.partition) == topology.n_labels:
            return Wald(topology=topology, weights=gs.ones(self.n_labels))

        target_and_gradient = self._proj_target_gradient(
            metric=metric,
            ambient_point=ambient_point,
            topology=topology,
        )

        n_splits = topology.n_splits
        bounds = self._get_bounds(n_splits)

        x0 = gs.ones(n_splits) * 0.5

        res = self._minimize(
            target_and_gradient, x0, bounds=bounds, **self.optimization_kwargs
        )

        if res.status != 0:
            raise ValueError("Projection failed!")

        x = [
            _x if self.btol < _x < 1 - self.btol else 0 if _x <= self.btol else 1
            for _x in res.x
        ]

        return Wald(topology=topology, weights=x)
