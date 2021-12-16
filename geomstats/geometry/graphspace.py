"""Graph Space."""

import numpy as np
from scipy.optimize import quadratic_assignment

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices, MatricesMetric


class _GraphSpace:
    r"""Class for the Graph Space.

    Graph Space to analyse populations of labelled and unlabelled graphs.
    The space is a quotient space obtained by applying the permutation
    action of nodes to the space of adjecency matrices.

    Points are represented by :math:`nodes \times nodes` adjecency matrices.

    Parameters
    ----------
    nodes : int
        Number of graph nodes
    p : int
        Dimension of euclidean parameter or label associated to a graph.

    References
    ----------
    ..[Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
              “Graph Space: Geodesic Principal Components for aPopulation of
              Network-valued Data.”
              Mox report 14, 2020.
              https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    """

    def __init__(self, n, p=None):
        self.nodes = n
        self.p = p
        self.adjmat = Matrices(self.nodes, self.nodes)

    def belongs(self, graph, atol=gs.atol):
        r"""Check if the matrix is an adjecency matrix associated to the
        graph with n nodes.

        Parameters
        ----------
        graph : array-like, shape=[..., n, n]
            Matrix to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,n]
            Boolean denoting if graph belongs to the space.
        """
        return self.adjmat.belongs(graph, atol=atol)

    def random_point(self, n_samples=1, bound=1.0):
        r"""Sample in Graph Space.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample in the tangent space.
            Optional, default: 1.

        Returns
        -------
        graph_samples : array-like, shape=[..., n, n]
            Points sampled in GraphSpace(n).
        """
        return self.adjmat.random_point(n_samples=n_samples, bound=bound)

    def permute(self, graph_to_permute, permutation):
        r"""Permutation action applied to graph observation.

        Parameters
        ----------
        graph_to_permute : array-like, shape=[..., n, n]
            input graphs to be permuted
        permutation: array-like, shape=[..., n]
            node permutations

        Returns
        -------
        graphs_permuted : array-like, shape=[..., n, n]
            Graphs permuted.
        """
        if len(graph_to_permute.shape) == 3:
            return gs.array(
                [graph_to_permute[i][:, p][p, :] for i, p in enumerate(permutation)]
            )
        else:
            return graph_to_permute[:, permutation][permutation, :]


class GraphSpaceMetric:
    """Quotient metric on the graph space.

    Parameters
    ----------
    nodes : int
        Number of nodes
    """

    def __init__(self, nodes):
        self.total_space_metric = MatricesMetric(nodes, nodes)
        self.nodes = nodes
        self.space = _GraphSpace(nodes)

    def dist(self, base_graph, graph_to_permute, matcher="ID"):
        """Compute the distance between two equivalence classes of
        adjecency matrices [Jain2009].

        Parameters
        ----------
        base_graph : array-like, shape=[..., n, n]
            First graph.
        graph_to_permute : array-like, shape=[..., n, n]
            Second graph to align to the first graph.
        matcher : selecting which matcher to use
            'FAQ': [Volgstain2015] Fast Quadratic Assignment
            only compatible with Frobenius metric.

        Returns
        -------
        distance : array-like, shape=[...,]
            distance between equivalence classes.

        References
        ----------
        ..[Jain2009]  Jain, B., Obermayer, K.
                  "Structure Spaces." Journal of Machine Learning Research 10.11 (2009).
                  https://www.jmlr.org/papers/v10/jain09a.html.
        """
        if matcher == "FAQ":
            perm = self.faq_matching(base_graph, graph_to_permute)
        if matcher == "ID":
            perm = self.id_matching(base_graph, graph_to_permute)
        return self.total_space_metric.dist(
            base_graph,
            self.space.permute(graph_to_permute=graph_to_permute, permutation=perm),
        )

    @staticmethod
    def faq_matching(base_graph, graph_to_permute):
        """Fast Quadratic Assignment for graph matching.

        Parameters
        ----------
        base_graph : array-like, shape=[..., n, n]
        First graph.
        graph_to_permute : array-like, shape=[..., n, n]
        Second graph to align.

        Returns
        -------
        permutation : array-like, shape=[...,n]
            node permutation indexes of the second graph.

        References
        ----------
        ..[Volgstain2015] Vogelstein JT, Conroy JM, Lyzinski V, Podrazik LJ,
                Kratzer SG, Harley ET, Fishkind DE, Vogelstein RJ, Priebe CE.
                “Fast approximate quadratic programming for graph matching.“
                PLoS One. 2015 Apr 17; doi: 10.1371/journal.pone.0121002.
        """
        l_base = len(base_graph.shape)
        l_obj = len(graph_to_permute.shape)
        if l_base == l_obj == 3:
            return gs.array(
                [
                    quadratic_assignment(x, y, options={"maximize": True})["col_ind"]
                    for x, y in zip(base_graph, graph_to_permute)
                ]
            )
        elif l_base == l_obj == 2:
            return quadratic_assignment(
                base_graph, graph_to_permute, options={"maximize": True}
            )["col_ind"]
        elif l_base < l_obj:
            return quadratic_assignment(
                np.stack([base_graph] * graph_to_permute.shape[0]),
                graph_to_permute,
                options={"maximize": True},
            )["col_ind"]
        else:
            raise ValueError(
                "The method can align a set of graphs to one graphs,"
                "but the single graphs should be passed as base_graph"
            )

    def id_matching(self, base_graph, graph_to_permute):
        """Identity matching.

        Parameters
        ----------
        base_graph : array-like, shape=[..., n, n]
        First graph.
        graph_to_permute : array-like, shape=[..., n, n]
        Second graph to align.

        Returns
        -------
        permutation : array-like, shape=[...,n]
            node permutation indexes of the second graph.
        """
        l_base = len(base_graph.shape)
        l_obj = len(graph_to_permute.shape)
        if l_base == l_obj == 3 or l_base < l_obj:
            return gs.repeat(
                [gs.arange(self.nodes)], repeats=len(graph_to_permute), axis=0
            )
        elif l_base == l_obj == 2:
            return gs.arange(self.nodes)
        else:
            raise (
                ValueError(
                    "The method can align a set of graphs to one graphs,"
                    "but the single graphs should be passed as base_graph"
                )
            )


class GraphSpace(_GraphSpace):
    """Class for Graph Space.

    Graphs are represented as adjecency matrices. The total space
    is an Euclidean space. The group action is the permutation
    node action applied to the total spacem to analyse set of
    node unlabelled graphs

    Parameters
    ----------
    nodes : int
        Number of nodes.

    p : int
        Dimension of the graph label or regressor.
    """

    def __init__(self, nodes, p=None):
        super(GraphSpace, self).__init__(nodes, p)
        self.metric = GraphSpaceMetric(nodes)
