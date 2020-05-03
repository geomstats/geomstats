"""The Agglomerative Hierarchical Clustering (AHC) on manifolds."""

from sklearn.cluster import AgglomerativeClustering

import geomstats.backend as gs


class AgglomerativeHierarchicalClustering(AgglomerativeClustering):
    """The Agglomerative Hierarchical Clustering on manifolds.

    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.
    distance : str or callable, default='euclidean'
        Distance used to compute the linkage. Can be 'euclidean', 'l1', 'l2',
        'manhattan', 'cosine', or 'precomputed'.
        If linkage is 'ward', only 'euclidean' is accepted.
        If 'precomputed', a distance matrix (instead of a similarity matrix)
        is needed as input for the fit method.
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
        https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/
        cluster/_agglomerative.py#L656
    """

    def __init__(self, n_clusters=2,
                 distance='euclidean',
                 memory=None,
                 connectivity=None,
                 compute_full_tree='auto',
                 linkage='average',
                 distance_threshold=None):

        if isinstance(distance, str):
            affinity = distance
        else:
            def affinity(data):
                n_samples = data.shape[0]
                affinity_matrix = gs.zeros([n_samples, n_samples])
                for i_sample in range(1, n_samples):
                    affinity_matrix[i_sample, :i_sample] = distance(
                        data[i_sample, ...], data[:i_sample, ...]).reshape(
                        i_sample)
                affinity_matrix += affinity_matrix.T
                return affinity_matrix

        super().__init__(
            n_clusters=n_clusters,
            affinity=affinity,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold)
