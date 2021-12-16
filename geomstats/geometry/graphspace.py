"""Graph Space."""

import geomstats.backend as gs
import numpy as np

from geomstats.geometry.matrices import Matrices, MatricesMetric
from scipy.optimize import quadratic_assignment


class _GraphSpace:
    r"""Class for the Graph Space.

    Graph Space to analyse populations of labelled and unlabelled graphs.
    The space is a quotient space obtained by applying the permutation
    action of nodes to the space of adjecency matrices.

    Points are represented by :math:`n \times n` adjecency matrices.

    Parameters
    ----------
    n : int
        Number of graph nodes
    p : int
        Dimension of euclidean parameter associated to a graph.

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
        belongs : array-like, shape=[...,]
            Boolean denoting if mat is belongs to the space.
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
        graph_to_permute : int
            Number of samples.
            Optional, default: 1.
        permutation

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points permuted.
        """
        return gs.array([mat[i][:, p][p, :]
                         for i, p in enumerate(permutation)])


class GraphSpaceMetric:
    """Quotient metric on the graph space.

    Parameters
    ----------
    n : int
        Number of nodes
    """

    def __init__(self, n):
        self.total_space_metric = MatricesMetric(n, n)
        self.nodes = n
        self.space = _GraphSpace(n)

    def dist(self, base_graph, graph_to_permute, matcher = 'ID'):
        """Compute the distance between two equivalence classes of
        adjecency matrices [Calissano2020].

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
            distance between the two equivalence classes.

        References
        ----------
        ..[Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
                  “Graph Space: Geodesic Principal Components for a
                  Population of Network-valued Data.”
                  Mox report 14, 2020.
                  https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
        ..[Volgstain2015] Vogelstein JT, Conroy JM, Lyzinski V, Podrazik LJ, Kratzer SG,
                Harley ET, Fishkind DE, Vogelstein RJ, Priebe CE.
                “Fast approximate quadratic programming for graph matching.“
                PLoS One. 2015 Apr 17;10(4):e0121002. doi: 10.1371/journal.pone.0121002.
        """
        if matcher == 'FAQ':
            perm = self.faq(base_graph, graph_to_permute)
        if matcher == 'ID':
            perm = self.id(base_graph, graph_to_permute)
        return self.total_space_metric.dist(base_graph,
                                            self.space.permute(mat=graph_to_permute,
                                                               permutation=perm))

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
            return gs.array([quadratic_assignment(x, y,
                                                  options={'maximize': True})['col_ind']
                             for x, y in zip(base_graph, graph_to_permute)])
        elif l_base == l_obj == 2:
            return quadratic_assignment(base_graph, graph_to_permute,
                                        options={'maximize': True})['col_ind']
        elif l_base < l_obj:
            return quadratic_assignment(np.stack([base_graph] *
                                                 graph_to_permute.shape[0]),
                                        graph_to_permute,
                                        options={'maximize': True})['col_ind']
        else:
            raise ValueError('The method can align a set of graphs to one graphs,'
                              'but the single graphs should be passed as base_graph')

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
            return gs.repeat([gs.arange(self.nodes)],
                             repeats=len(graph_to_permute), axis=0)
        elif l_base == l_obj == 2:
            return gs.arange(self.nodes)
        else:
            raise (ValueError('The method can align a set of graphs to one graphs,'
                              'but the single graphs should be passed as base_graph'))

class GraphSpace(_GraphSpace):
    """Class for the n-dimensional hypersphere.

    Class for the n-dimensional hypersphere embedded in the
    (n+1)-dimensional Euclidean space.

    By default, points are parameterized by their extrinsic
    (n+1)-coordinates. For dimensions 1 and 2, this can be changed with the
    `default_coords_type` parameter. For dimensions 1 (the circle),
    the intrinsic coordinates correspond angles in radians, with 0. mapping
    to point [1., 0.]. For dimension 2, the intrinsic coordinates are the
    spherical coordinates from the north pole, i.e. where angles [0.,
    0.] correspond to point [0., 0., 1.].

    Parameters
    ----------
    dim : int
        Dimension of the hypersphere.

    default_coords_type : str, {'extrinsic', 'intrinsic'}
        Type of representation for dimensions 1 and 2.
    """

    def __init__(self, n, p=None):
        super(GraphSpace, self).__init__(n, p)
        self.metric = GraphSpaceMetric(n)
