"""Graph Space.

Lead author: Anna Calissano.
"""

import itertools
from abc import ABC, abstractmethod

import scipy

import geomstats.backend as gs
from geomstats.errors import check_parameter_accepted_values
from geomstats.geometry.group_action import PermutationAction
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.stratified.quotient import (
    Aligner,
    BaseAlignerAlgorithm,
    QuotientMetric,
)
from geomstats.vectorization import check_is_batch, get_batch_shape


class _GraphSpaceAlignerAlgorithm(BaseAlignerAlgorithm, ABC):
    """Base class for graph space numerical aligner.

    Attributes
    ----------
    perm_ : array-like, shape=[..., n_nodes]
        Node permutations where in position i we have the value j meaning
        the node i should be permuted with node j.
    """

    def __init__(self):
        # TODO: rename to `opt_group_elem` or `opt_elem`?
        # Send it to `Aligner` (would change `Aligner.align`)?
        self.perm_ = None


class FAQAligner(_GraphSpaceAlignerAlgorithm):
    """Fast Quadratic Assignment for graph matching (or network alignment).

    References
    ----------
    .. [Vogelstein2015] Vogelstein JT, Conroy JM, Lyzinski V, Podrazik LJ,
        Kratzer SG, Harley ET, Fishkind DE, Vogelstein RJ, Priebe CE.
        “Fast approximate quadratic programming for graph matching.“
        PLoS One. 2015 Apr 17; doi: 10.1371/journal.pone.0121002.
    """

    def _align_single(self, point, base_point):
        """Get optimal element of the group."""
        return gs.linalg.quadratic_assignment(
            base_point, point, options={"maximize": True}
        )

    def align(self, total_space, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        total_space : PointSet
            PointSet with quotient structure.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.
        base_point : array-like, shape=[..., n_nodes, n_nodes]
            Reference graph.

        Returns
        -------
        aligned_point : array-like, shape=[..., n_nodes, n_nodes]
            Aligned graph.
        """
        is_batch = check_is_batch(total_space.point_ndim, point, base_point)
        if is_batch:
            if point.ndim != base_point.ndim:
                point, base_point = gs.broadcast_arrays(point, base_point)
            self.perm_ = gs.array(
                [
                    self._align_single(point_, base_point_)
                    for point_, base_point_ in zip(point, base_point)
                ]
            )
        else:
            self.perm_ = gs.array(self._align_single(point, base_point))

        return total_space.group_action.act(self.perm_, point)


class IDAligner(_GraphSpaceAlignerAlgorithm):
    """Identity alignment.

    The identity alignment is not performing any matching but returning the nodes in
    their original position. This alignment can be selected when working with labelled
    graphs.
    """

    def align(self, total_space, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        total_space : PointSet
            PointSet with quotient structure.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.
        base_point : array-like, shape=[..., n_nodes, n_nodes]
            Reference graph.

        Returns
        -------
        aligned_point : array-like, shape=[..., n_nodes, n_nodes]
            Aligned graph.
        """
        n_nodes = total_space.n_nodes
        perm = gs.array(list(range(n_nodes)))
        if base_point.ndim > point.ndim:
            point = gs.broadcast_to(point, base_point.shape)

        if point.ndim > 2:
            perm = gs.broadcast_to(perm, point.shape[:-2] + (n_nodes,))

        self.perm_ = perm

        return gs.copy(point)


class ExhaustiveAligner(_GraphSpaceAlignerAlgorithm):
    """Brute force exact alignment.

    Exact Alignment obtained by exploring the whole permutation group.

    Notes
    -----
    Not recommended for large `n_nodes`.
    """

    def __init__(self):
        super().__init__()
        self._all_perms = None
        self._n_nodes = None

    def _get_all_perms(self, n_nodes):
        if self._all_perms is None or self._n_nodes != n_nodes:
            self._n_nodes = n_nodes
            self._all_perms = gs.array(
                list(itertools.permutations(range(n_nodes), n_nodes))
            )

        return self._all_perms

    def _align_single(self, total_space, point, base_point):
        """Get optimal element of the group."""
        perms = self._get_all_perms(total_space.n_nodes)

        aligned_points = total_space.group_action.act(perms, point)
        dists = total_space.metric.dist(base_point, aligned_points)
        return perms[gs.argmin(dists)]

    def align(self, total_space, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        total_space : PointSet
            PointSet with quotient structure.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.
        base_point : array-like, shape=[..., n_nodes, n_nodes]
            Reference graph.

        Returns
        -------
        aligned_point : array-like, shape=[..., n_nodes, n_nodes]
            Aligned graph.
        """
        total_space = total_space
        is_batch = check_is_batch(total_space.point_ndim, point, base_point)
        if is_batch:
            if point.ndim != base_point.ndim:
                point, base_point = gs.broadcast_arrays(point, base_point)
            self.perm_ = gs.array(
                [
                    self._align_single(total_space, point_, base_point_)
                    for point_, base_point_ in zip(point, base_point)
                ]
            )
        else:
            self.perm_ = gs.array(self._align_single(total_space, point, base_point))

        return total_space.group_action.act(self.perm_, point)


class _BasePointToGeodesicAligner(ABC):
    """Base class for point to geodesic aligner.

    Attributes
    ----------
    perm_ : array-like, shape=[..., n_nodes]
        Node permutations where in position i we have the value j meaning
        the node i should be permuted with node j.
    """

    def __init__(self):
        self.perm_ = None

    @abstractmethod
    def align(self, total_space, geodesic, graph_to_permute):
        """Class for the Alignment of the geodesic with respect to a point."""

    @abstractmethod
    def dist(self, total_space, geodesic, graph_to_permute):
        """Class to compute distance between the geodesic with respect to a point."""
        # TODO: require as abstract?


class PointToGeodesicAligner(_BasePointToGeodesicAligner):
    """Class for the Alignment of the points with respect to a geodesic.

    Implementing the algorithm in [Huckemann2010]_ to select an optimal alignment to a
    point with respect to a geodesic. The algorithm sample discrete set of n_points
    along the geodesic between [s_min, s_max] and find the permutation that gets closer
    to the datapoints along the geodesic.

    Parameters
    ----------
    s_min : float
        Minimum value of the domain to sample along the geodesics.
    s_max : float
        Minimum value of the domain to sample along the geodesics.
    n_points: int
        Number of points to sample between s_min and s_max.

    References
    ----------
    .. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
        “Graph Space: Geodesic Principal Components for a Population of
        Network-valued Data.” Mox report 14, 2020.
        https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    .. [Huckemann2010] Huckemann, S., Hotz, T., Munk, A.
        "Intrinsic shape analysis: Geodesic PCA for Riemannian manifolds modulo
        isometric Lie group actions." Statistica Sinica, 1-58, 2010.
    """

    def __init__(self, s_min, s_max, n_points=10):
        # TODO: rename n_points to n_grid
        super().__init__()
        self.s_min = s_min
        self.s_max = s_max
        self.n_points = n_points
        self._s = None

    def __setattr__(self, attr_name, value):
        """Set attributes."""
        if attr_name in ["s_min", "s_max", "n_points"]:
            self._s = None

        return object.__setattr__(self, attr_name, value)

    def _discretize_s(self):
        """Compute the domain distretization."""
        return gs.linspace(self.s_min, self.s_max, num=self.n_points)

    @property
    def s(self):
        """Save the domain discretization."""
        if self._s is None:
            self._s = self._discretize_s()

        return self._s

    def _get_geodesic_s(self, geodesic):
        """Evaluate the geodesic in s."""
        return geodesic(self.s)

    def _compute_dists(self, total_space, geodesic, point):
        geodesic_s = self._get_geodesic_s(geodesic)
        geo_shape = (self.n_points,) + total_space.shape

        geo_batch_shape = get_batch_shape(3, geodesic_s)
        point_batch_shape = get_batch_shape(2, point)
        if point_batch_shape or geo_batch_shape:
            batch_shape = point_batch_shape or geo_batch_shape
        else:
            batch_shape = ()

        point = gs.broadcast_to(
            point, (self.n_points,) + batch_shape + total_space.shape
        )
        if point_batch_shape:
            point = gs.moveaxis(point, 0, -3)

        if batch_shape and not geo_batch_shape:
            geodesic_s = gs.broadcast_to(geodesic_s, batch_shape + geo_shape)

        flat_point = gs.reshape(point, (-1,) + total_space.shape)
        flat_geodesic_s = gs.reshape(geodesic_s, (-1,) + total_space.shape)

        aligned_flat_points = total_space.aligner.align(flat_geodesic_s, flat_point)
        flat_dists = total_space.metric.dist(flat_geodesic_s, aligned_flat_points)

        perm_ = total_space.aligner.aligner.perm_
        total_space.aligner.aligner.perm_ = gs.reshape(
            perm_, batch_shape + (self.n_points, total_space.n_nodes)
        )

        dists = gs.reshape(
            flat_dists,
            batch_shape + (self.n_points,),
        )
        min_dists_idx = gs.argmin(dists, axis=-1)

        aligned_points = gs.reshape(
            aligned_flat_points,
            batch_shape + geo_shape,
        )

        return dists, min_dists_idx, aligned_points

    def dist(self, total_space, geodesic, point):
        """Compute the distance between the geodesic and the point.

        Parameters
        ----------
        total_space : GraphSpace
        geodesic : function
            Geodesic function in GraphSpace.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        dist : array-like, shape=[..., n_nodes]
            Distance between the point and the geodesic.
        """
        dists, min_dists_idx, _ = self._compute_dists(total_space, geodesic, point)

        slc = []
        for n in dists.shape[:-1]:
            slc.append(gs.arange(n))
        slc.append(min_dists_idx)
        return dists[tuple(slc)]

    def align(self, total_space, geodesic, point):
        """Align the graph to the geodesic.

        Parameters
        ----------
        total_space : GraphSpace
        geodesic : function
            Geodesic function in GraphSpace.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        permuted_graph : array-like, shape=[..., n_nodes]
            Permuted graph as to be aligned with respect to the geodesic.
        """
        _, min_dists_idx, aligned_points = self._compute_dists(
            total_space, geodesic, point
        )
        slc = []
        for n in aligned_points.shape[:-3]:
            slc.append(gs.arange(n))
        slc.extend([min_dists_idx, slice(None), slice(None)])
        return aligned_points[tuple(slc)]


class _GeodesicToPointAligner(_BasePointToGeodesicAligner):
    def __init__(self, method="BFGS", *, save_opt_res=False):
        super().__init__()

        self.method = method
        self.save_opt_res = save_opt_res

        self.opt_results_ = None

    def _objective(self, s, space, graph, geodesic):
        point = geodesic(s)
        dist = space.metric.dist(point, graph)

        return dist

    def _compute_dists(self, space, geodesic, graph_to_permute):
        n_points = self._get_n_points(graph_to_permute)

        if n_points == 1:
            graph_to_permute = gs.expand_dims(graph_to_permute, axis=0)

        perms = []
        min_dists = []
        opt_results = []
        for graph in graph_to_permute:
            s0 = 0.0
            res = scipy.optimize.minimize(
                self._objective,
                x0=s0,
                args=(space, graph, geodesic),
                method=self.method,
            )
            perms.append(space.metric.perm_[0])
            min_dists.append(res.fun)

            opt_results.append(res)

        if self.save_opt_res:
            self.opt_results_ = opt_results

        return gs.array(min_dists), gs.array(perms), n_points

    def dist(self, space, geodesic, graph_to_permute):
        dists, _, _ = self._compute_dists(space, geodesic, graph_to_permute)

        return dists

    def align(self, space, geodesic, graph_to_permute):
        _, perms, n_points = self._compute_dists(space, geodesic, graph_to_permute)

        new_graph = self._permute(space, graph_to_permute, perms)
        self.perm_ = perms[0] if n_points == 1 else perms

        return new_graph


class GraphSpace(Matrices):
    r"""Class for the Graph Space.

    Graph Space to analyse populations of labelled and unlabelled graphs.
    The space focuses on graphs with scalar euclidean attributes on nodes and edges,
    with a finite number of nodes and both directed and undirected edges.
    For undirected graphs, use symmetric adjacency matrices. The space is a quotient
    space obtained by applying the permutation action of nodes to the space
    of adjacency matrices. Notice that for computation reasons the module works with
    both the `gs.array` representation of graph and the `GraphPoint` representation.

    Points are represented by :math:`nodes \times nodes` adjacency matrices.
    Both the array input and the Graph Point type input work.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes
    total_space : space
        Total Space before applying the permutation action. Default: Euclidean Space.

    References
    ----------
    .. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
        “Graph Space: Geodesic Principal Components for a Population of
        Network-valued Data.” Mox report 14, 2020.
        https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    .. [Jain2009] Jain, B., Obermayer, K.
        "Structure Spaces." Journal of Machine Learning Research, 10(11), 2009.
        https://www.jmlr.org/papers/volume10/jain09a/jain09a.pdf
    """

    def __init__(self, n_nodes, equip=True):
        self.n_nodes = n_nodes
        super().__init__(n_nodes, n_nodes, equip=equip)

        self._quotient_map = {
            (MatricesMetric, PermutationAction): (
                GraphSpaceAligner,
                GraphSpaceQuotientMetric,
            )
        }

    def new(self, equip=True):
        """Create manifold with same parameters."""
        return GraphSpace(n_nodes=self.n_nodes, equip=equip)

    def equip_with_group_action(self, group_action="permutations"):
        """Equip manifold with group action."""
        if group_action == "permutations":
            group_action = PermutationAction()

        return super().equip_with_group_action(group_action)

    def equip_with_quotient_structure(self):
        """Equip manifold with quotient structure.

        Creates attributes `quotient` and `aligner`.

        NB: `aligner` instead of `bundle` because total space does not
        have fiber bundle structure due to the nature of the group actions
        on this space.
        """
        self._check_equip_with_quotient_structure()

        key = type(self.metric), type(self.group_action)

        out = self._quotient_map.get(key, None)
        if out is None:
            raise ValueError(f"No mapping for key: {key}")
        Aligner, QuotientMetric_ = out

        self.aligner = Aligner(self)

        self.quotient = self.new(equip=False)
        self.quotient.equip_with_metric(QuotientMetric_, total_space=self)

        return self.quotient


class GraphSpaceAligner(Aligner):
    """Graph space aligner."""

    MAP_ALIGNER = {
        "ID": IDAligner,
        "FAQ": FAQAligner,
        "exhaustive": ExhaustiveAligner,
    }
    # TODO: set register strategy?
    # TODO: rename things as numerical_aligner

    def __init__(self, total_space, aligner_algorithm=None):
        if aligner_algorithm is None:
            aligner_algorithm = self._set_default_aligner_algorithm()

        super().__init__(total_space=total_space, aligner_algorithm=aligner_algorithm)
        self.point_to_geodesic_aligner = None

    def _set_default_aligner_algorithm(self):
        return self.set_aligner_algorithm("FAQ")

    def set_aligner_algorithm(self, aligner, **kwargs):
        """Set the aligning strategy.

        Graph Space metric relies on alignment. In this module we propose the
        identity matching, the FAQ graph matching by [Vogelstein2015]_, and
        exhaustive aligner which explores the whole permutation group.

        Parameters
        ----------
        aligner : str
            'ID': Identity
            'FAQ': Fast Quadratic Assignment - only compatible with Frobenius norm
            'exhaustive': all group exhaustive search
        """
        if isinstance(aligner, str):
            check_parameter_accepted_values(
                aligner, "aligner", list(self.MAP_ALIGNER.keys())
            )

            aligner = self.MAP_ALIGNER.get(aligner)(**kwargs)

        self.aligner = aligner
        return self.aligner

    @property
    def perm_(self):
        """Permutation of nodes after alignment.

        Node permutations where in position i we have the value j meaning
        the node i should be permuted with node j.
        """
        # TODO: rename if also renamed above
        return self.aligner.perm_

    def set_point_to_geodesic_aligner(self, aligner, **kwargs):
        """Set the alignment between a point and a geodesic.

        Following the geodesic to point alignment in [Calissano2020]_ and
        [Huckemann2010]_, this function defines the parameters [s_min, s_max] and
        the number of points to sample in the domain.

        Parameters
        ----------
        s_min : float
            Minimum value of the domain to sample along the geodesics.
        s_max : float
            Minimum value of the domain to sample along the geodesics.
        n_points: int
            Number of points to sample between s_min and s_max.
        """
        if aligner == "default":
            kwargs.setdefault("s_min", -1.0)
            kwargs.setdefault("s_max", 1.0)
            kwargs.setdefault("n_points", 10)
            aligner = PointToGeodesicAligner(**kwargs)

        self.point_to_geodesic_aligner = aligner
        return self.point_to_geodesic_aligner

    def align_point_to_geodesic(self, geodesic, point):
        """Align point to a geodesic.

        Using the selected alignment technique, it returns the aligned
        point as optimally aligned to the geodesic.

        Parameters
        ----------
        geodesic : function
            Geodesic.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        permuted_graph: list, shape = [..., n_nodes, n_nodes]
        """
        if self.point_to_geodesic_aligner is None:
            raise UnboundLocalError(
                "Set point to geodesic aligner first (e.g. "
                "`metric.set_point_to_geodesic_aligner('default', "
                "s_min=-1., s_max=1.)`)"
            )
        return self.point_to_geodesic_aligner.align(self._total_space, geodesic, point)


class GraphSpaceQuotientMetric(QuotientMetric):
    r"""Class for the Graph Space Metric.

    Every metric :math:`d: X \times X \rightarrow \mathbb{R}` on the total space of
    adjacency matrices can descend onto the quotient space as a pseudo-metric:
    :math:`d([x_1],[x_2]) = min_{t\in T} d_X(x_1, t^Tx_2t)`. The metric relies on the
    total space metric and an alignment procedure, i.e., Graph Matching or Networks
    alignment procedure. Metric, alignment, geodesics, and alignment with respect to
    a geodesic are defined. By default, the alignment is the identity and the total
    space metric is the Frobenious norm.

    Parameters
    ----------
    space : GraphSpace
        GraphSpace object.

    References
    ----------
    .. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
        “Graph Space: Geodesic Principal Components for a Population of
        Network-valued Data.” Mox report 14, 2020.
        https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    .. [Jain2009]  Jain, B., Obermayer, K.
        "Structure Spaces." Journal of Machine Learning Research 10.11 (2009).
        https://www.jmlr.org/papers/v10/jain09a.html.
    """
