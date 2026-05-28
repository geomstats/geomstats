"""The Agglomerative Hierarchical Clustering (AHC) on manifolds."""

import warnings

import numpy as np
import sklearn.cluster._agglomerative as ca
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._agglomerative import (
    METRIC_MAPPING64,
    DistanceMetric,
    IntFloatDict,
    _fix_connectivity,
    _hierarchical,
    _single_linkage_tree,
    heapify,
    heappop,
    heappush,
    paired_distances,
)

import geomstats.backend as gs

from ._sklearn import ObjectValidationMixin, _enable_array_dispatch

_enable_array_dispatch()


class AgglomerativeHierarchicalClustering(
    ObjectValidationMixin, AgglomerativeClustering
):
    """The Agglomerative Hierarchical Clustering on manifolds.

    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.
    memory : str or object, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.
    connectivity : array-like or callable, default=None
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix. Default is None, i.e, the
        hierarchical clustering algorithm is unstructured.
    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at n_clusters. This is useful
        to decrease computation time if the number of clusters is not small
        compared to the number of samples. This option is useful only when
        specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is 'auto', which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, 'auto' is equivalent to `False`.
    linkage : {'ward', 'complete', 'average', 'single'}, default='average'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - average uses the average of the distances of each observation of
          the two sets.
        - complete or maximum linkage uses the maximum distances between
          all observations of the two sets.
        - single uses the minimum of the distances between all observations
          of the two sets.
        - ward minimizes the variance of the clusters being merged.
          It works for the 'euclidean' distance only.
    distance_threshold : float, default=None
        The linkage distance threshold above which, clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.
    labels_ : ndarray, shape=[...,]
        Cluster labels for each point.
    n_leaves_ : int
        Number of leaves in the hierarchical tree.
    n_connected_components_ : int
        The estimated number of connected components in the graph.
    children_ : array-like, shape=[n_samples-1, 2]
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    References
    ----------
    This algorithm uses the scikit-learn library:
    https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/cluster/_agglomerative.py#L656
    """

    _object_validation_methods = {
        "fit",
    }

    def __init__(
        self,
        space,
        n_clusters=2,
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="average",
        distance_threshold=None,
    ):
        def affinity(data):
            n_samples = data.shape[0]
            affinity_matrix = gs.zeros([n_samples, n_samples])
            for i_sample in range(1, n_samples):
                affinity_matrix[i_sample, :i_sample] = self.space.metric.dist(
                    data[i_sample, ...], data[:i_sample, ...]
                ).reshape(i_sample)
            affinity_matrix = affinity_matrix + affinity_matrix.T
            return affinity_matrix

        self.space = space

        self._skip_validation = gs.__name__.endswith("pytorch")
        if self._skip_validation:
            self._set_validation()

        super().__init__(
            n_clusters=n_clusters,
            metric=affinity,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
        )

    def _set_validation(self):
        self._object_validation_modules = (ca,)
        self._object_validation_names = ("linkage_tree",)
        self._object_validation_values = (linkage_tree,)


# TODO (L):
def linkage_tree(
    X,
    connectivity=None,
    n_clusters=None,
    linkage="complete",
    affinity="euclidean",
    return_distance=False,
):
    """Linkage agglomerative clustering based on a Feature matrix.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account some topological
    structure between samples.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix representing `n_samples` samples to be clustered.

    connectivity : sparse matrix, default=None
        Connectivity matrix. Defines for each sample the neighboring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is `None`, i.e, the Ward algorithm is unstructured.

    n_clusters : int, default=None
        Stop early the construction of the tree at `n_clusters`. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. In this case, the
        complete tree is not computed, thus the 'children' output is of
        limited use, and the 'parents' output should rather be used.
        This option is valid only when specifying a connectivity matrix.

    linkage : {"average", "complete", "single"}, default="complete"
        Which linkage criteria to use. The linkage criterion determines which
        distance to use between sets of observation.
            - "average" uses the average of the distances of each observation of
              the two sets.
            - "complete" or maximum linkage uses the maximum distances between
              all observations of the two sets.
            - "single" uses the minimum of the distances between all
              observations of the two sets.

    affinity : str or callable, default='euclidean'
        Which metric to use. Can be 'euclidean', 'manhattan', or any
        distance known to paired distance (see metric.pairwise).

    return_distance : bool, default=False
        Whether or not to return the distances between the clusters.

    Returns
    -------
    children : ndarray of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    n_connected_components : int
        The number of connected components in the graph.

    n_leaves : int
        The number of leaves in the tree.

    parents : ndarray of shape (n_nodes, ) or None
        The parent of each node. Only returned when a connectivity matrix
        is specified, elsewhere 'None' is returned.

    distances : ndarray of shape (n_nodes-1,)
        Returned when `return_distance` is set to `True`.

        distances[i] refers to the distance between children[i][0] and
        children[i][1] when they are merged.

    See Also
    --------
    ward_tree : Hierarchical clustering with ward linkage.
    """
    # TODO (L): backend issue
    # X = np.asarray(X)
    # if X.ndim == 1:
    #     X = np.reshape(X, (-1, 1))
    n_samples, n_features = X.shape

    linkage_choices = {
        "complete": _hierarchical.max_merge,
        "average": _hierarchical.average_merge,
        "single": None,
    }  # Single linkage is handled differently
    try:
        join_func = linkage_choices[linkage]
    except KeyError as e:
        raise ValueError(
            "Unknown linkage option, linkage should be one of %s, but %s was given"
            % (linkage_choices.keys(), linkage)
        ) from e

    if affinity == "cosine" and np.any(~np.any(X, axis=1)):
        raise ValueError("Cosine affinity cannot be used when X contains zero vectors")

    if connectivity is None:
        from scipy.cluster import hierarchy  # imports PIL

        if n_clusters is not None:
            warnings.warn(
                (
                    "Partial build of the tree is implemented "
                    "only for structured clustering (i.e. with "
                    "explicit connectivity). The algorithm "
                    "will build the full tree and only "
                    "retain the lower branches required "
                    "for the specified number of clusters"
                ),
                stacklevel=2,
            )

        if affinity == "precomputed":
            # for the linkage function of hierarchy to work on precomputed
            # data, provide as first argument an ndarray of the shape returned
            # by sklearn.metrics.pairwise_distances.
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    f"Distance matrix should be square, got matrix of shape {X.shape}"
                )
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
        elif affinity == "l2":
            # Translate to something understood by scipy
            affinity = "euclidean"
        elif affinity in ("l1", "manhattan"):
            affinity = "cityblock"
        elif callable(affinity):
            X = affinity(X)
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
        if (
            linkage == "single"
            and affinity != "precomputed"
            and not callable(affinity)
            and affinity in METRIC_MAPPING64
        ):
            # We need the fast cythonized metric from neighbors
            dist_metric = DistanceMetric.get_metric(affinity)

            # The Cython routines used require contiguous arrays
            X = np.ascontiguousarray(X, dtype=np.double)

            mst = _hierarchical.mst_linkage_core(X, dist_metric)
            # Sort edges of the min_spanning_tree by weight
            mst = mst[np.argsort(mst.T[2], kind="mergesort"), :]

            # Convert edge list into standard hierarchical clustering format
            out = _hierarchical.single_linkage_label(mst)
        else:
            out = hierarchy.linkage(X, method=linkage, metric=affinity)

        # TODO (L): backend issue
        # children_ = out[:, :2].astype(int, copy=False)
        children_ = gs.cast(out[:, :2], dtype=gs.int64)

        if return_distance:
            distances = out[:, 2]
            return children_, 1, n_samples, None, distances
        return children_, 1, n_samples, None

    connectivity, n_connected_components = _fix_connectivity(
        X, connectivity, affinity=affinity
    )
    connectivity = connectivity.tocoo()
    # Put the diagonal to zero
    diag_mask = connectivity.row != connectivity.col
    connectivity.row = connectivity.row[diag_mask]
    connectivity.col = connectivity.col[diag_mask]
    connectivity.data = connectivity.data[diag_mask]
    del diag_mask

    if affinity == "precomputed":
        distances = X[connectivity.row, connectivity.col].astype(np.float64, copy=False)
    else:
        # FIXME We compute all the distances, while we could have only computed
        # the "interesting" distances
        distances = paired_distances(
            X[connectivity.row], X[connectivity.col], metric=affinity
        )
    connectivity.data = distances

    if n_clusters is None:
        n_nodes = 2 * n_samples - 1
    else:
        assert n_clusters <= n_samples
        n_nodes = 2 * n_samples - n_clusters

    if linkage == "single":
        return _single_linkage_tree(
            connectivity,
            n_samples,
            n_nodes,
            n_clusters,
            n_connected_components,
            return_distance,
        )

    if return_distance:
        distances = np.empty(n_nodes - n_samples)
    # create inertia heap and connection matrix
    A = np.empty(n_nodes, dtype=object)
    inertia = list()

    # LIL seems to the best format to access the rows quickly,
    # without the numpy overhead of slicing CSR indices and data.
    connectivity = connectivity.tolil()
    # We are storing the graph in a list of IntFloatDict
    for ind, (data, row) in enumerate(zip(connectivity.data, connectivity.rows)):
        A[ind] = IntFloatDict(
            np.asarray(row, dtype=np.intp), np.asarray(data, dtype=np.float64)
        )
        # We keep only the upper triangular for the heap
        # Generator expressions are faster than arrays on the following
        inertia.extend(
            _hierarchical.WeightedEdge(d, ind, r) for r, d in zip(row, data) if r < ind
        )
    del connectivity

    heapify(inertia)

    # prepare the main fields
    parent = np.arange(n_nodes, dtype=np.intp)
    used_node = np.ones(n_nodes, dtype=np.intp)
    children = []

    # recursive merge loop
    for k in range(n_samples, n_nodes):
        # identify the merge
        while True:
            edge = heappop(inertia)
            if used_node[edge.a] and used_node[edge.b]:
                break
        i = edge.a
        j = edge.b

        if return_distance:
            # store distances
            distances[k - n_samples] = edge.weight

        parent[i] = parent[j] = k
        children.append((i, j))
        # Keep track of the number of elements per cluster
        n_i = used_node[i]
        n_j = used_node[j]
        used_node[k] = n_i + n_j
        used_node[i] = used_node[j] = False

        # update the structure matrix A and the inertia matrix
        # a clever 'min', or 'max' operation between A[i] and A[j]
        coord_col = join_func(A[i], A[j], used_node, n_i, n_j)
        for col, d in coord_col:
            A[col].append(k, d)
            # Here we use the information from coord_col (containing the
            # distances) to update the heap
            heappush(inertia, _hierarchical.WeightedEdge(d, k, col))
        A[k] = coord_col
        # Clear A[i] and A[j] to save memory
        A[i] = A[j] = 0

    # Separate leaves in children (empty lists up to now)
    n_leaves = n_samples

    # # return numpy array for efficient caching
    children = np.array(children)[:, ::-1]

    if return_distance:
        return children, n_connected_components, n_leaves, parent, distances
    return children, n_connected_components, n_leaves, parent
