"""Graph Space.

Lead author: Anna Calissano.

References
----------
.. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
    “Graph Space: Geodesic Principal Components for a Population of
    Network-valued Data.” Mox report 14, 2020.
    https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
"""

import itertools
from abc import ABC, abstractmethod

import geomstats.backend as gs
from geomstats.errors import check_parameter_accepted_values
from geomstats.geometry.fiber_bundle import AlignerAlgorithm
from geomstats.geometry.group_action import PermutationAction
from geomstats.geometry.manifold import register_quotient
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.stratified.quotient import Aligner, QuotientMetric
from geomstats.numerics.optimization import ScipyMinimize
from geomstats.vectorization import check_is_batch, get_batch_shape


class GraphSpaceAlignerAlgorithm(AlignerAlgorithm, ABC):
    """Base class for graph space numerical aligner.

    Attributes
    ----------
    total_space : GraphSpace
        Set with quotient structure.
    perm_ : array-like, shape=[..., n_nodes]
        Node permutations where in position i we have the value j meaning
        the node i should be permuted with node j.
    """

    def __init__(self, total_space):
        super().__init__(total_space)
        self.perm_ = None

    def _get_opt_perm(self, point, base_point):
        """Get optimal element of the group.

        Parameters
        ----------
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.
        base_point : array-like, shape=[..., n_nodes, n_nodes]
            Base graph.

        Returns
        -------
        perm : array-like, shape=[..., n_nodes]
            Optimal permutation group element.
        """
        is_batch = check_is_batch(self._total_space.point_ndim, point, base_point)
        if is_batch:
            if point.ndim != base_point.ndim:
                point, base_point = gs.broadcast_arrays(point, base_point)
            return gs.stack(
                [
                    self._get_opt_perm_single(point_, base_point_)
                    for point_, base_point_ in zip(point, base_point)
                ]
            )
        return self._get_opt_perm_single(point, base_point)

    def align(self, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.
        base_point : array-like, shape=[..., n_nodes, n_nodes]
            Reference graph.

        Returns
        -------
        aligned_point : array-like, shape=[..., n_nodes, n_nodes]
            Aligned graph.
        """
        self.perm_ = self._get_opt_perm(point, base_point)
        return self._total_space.group_action(self.perm_, point)


class FAQAligner(GraphSpaceAlignerAlgorithm):
    """Fast Quadratic Assignment for graph matching (or network alignment).

    References
    ----------
    .. [Vogelstein2015] Vogelstein JT, Conroy JM, Lyzinski V, Podrazik LJ,
        Kratzer SG, Harley ET, Fishkind DE, Vogelstein RJ, Priebe CE.
        “Fast approximate quadratic programming for graph matching.“
        PLoS One. 2015 Apr 17; doi: 10.1371/journal.pone.0121002.
    """

    def _get_opt_perm_single(self, point, base_point):
        """Get optimal element of the group.

        Parameters
        ----------
        point : array-like, shape=[n_nodes, n_nodes]
            Graph to align.
        base_point : array-like, shape=[n_nodes, n_nodes]
            Base graph.

        Returns
        -------
        perm : array-like, shape=[n_nodes]
            Optimal permutation group element.
        """
        return gs.array(
            gs.linalg.quadratic_assignment(
                base_point, point, options={"maximize": True}
            )
        )


class ExhaustiveAligner(GraphSpaceAlignerAlgorithm):
    """Brute force exact alignment.

    Exact Alignment obtained by exploring the whole permutation group.

    Parameters
    ----------
    total_space : GraphSpace
        Set with quotient structure.

    Notes
    -----
    Not recommended for large `n_nodes`.
    """

    def __init__(self, total_space):
        super().__init__(total_space)
        n_nodes = total_space.n_nodes
        self._perms = gs.array(list(itertools.permutations(range(n_nodes), n_nodes)))

    def _get_opt_perm_single(self, point, base_point):
        """Get optimal element of the group.

        Parameters
        ----------
        point : array-like, shape=[n_nodes, n_nodes]
            Graph to align.
        base_point : array-like, shape=[n_nodes, n_nodes]
            Base graph.

        Returns
        -------
        perm : array-like, shape=[n_nodes]
            Optimal permutation group element.
        """
        orbit = self._total_space.group_action(self._perms, point)
        sdists = self._total_space.metric.squared_dist(base_point, orbit)
        return self._perms[gs.argmin(sdists)]


class PointToGeodesicAlignerBase(ABC):
    """Base class for point to geodesic aligner.

    Parameters
    ----------
    total_space : GraphSpace
        Set with quotient structure.

    Attributes
    ----------
    perm_ : array-like, shape=[..., n_nodes]
        Node permutations where in position i we have the value j meaning
        the node i should be permuted with node j.
    """

    def __init__(self, total_space):
        self._total_space = total_space
        self.perm_ = None

    @abstractmethod
    def align(self, geodesic, point):
        """Class for the alignment of the geodesic with respect to a point."""

    @abstractmethod
    def dist(self, geodesic, point):
        """Class to compute distance between the geodesic with respect to a point."""


class PointToGeodesicAligner(PointToGeodesicAlignerBase):
    """Class for the alignment of the points with respect to a geodesic.

    Implementing the algorithm in [Huckemann2010]_ to select an optimal alignment to a
    point with respect to a geodesic. The algorithm sample discrete set of n_points
    along the geodesic between [s_min, s_max] and find the permutation that gets closer
    to the datapoints along the geodesic.

    Parameters
    ----------
    total_space : GraphSpace
        Set with quotient structure.
    s_min : float
        Minimum value of the domain to sample along the geodesics.
    s_max : float
        Minimum value of the domain to sample along the geodesics.
    n_grid: int
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

    def __init__(self, total_space, s_min=0.0, s_max=1.0, n_grid=10):
        super().__init__(total_space)
        self.s_min = s_min
        self.s_max = s_max
        self.n_grid = n_grid
        self._s_grid = None

    def __setattr__(self, attr_name, value):
        """Set attributes."""
        if attr_name in ["s_min", "s_max", "n_points"]:
            self._s_grid = None

        return object.__setattr__(self, attr_name, value)

    def _discretize_s(self):
        """Compute the domain distretization."""
        return gs.linspace(self.s_min, self.s_max, num=self.n_grid)

    @property
    def s_grid(self):
        """Save the domain discretization."""
        if self._s_grid is None:
            self._s_grid = self._discretize_s()

        return self._s_grid

    def _compute_dists(self, geodesic, point):
        """Compute the distance between the geodesic and the point.

        Parameters
        ----------
        geodesic : function
            Geodesic function in GraphSpace.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        dists : array-like, shape=[..., n_grid]
        min_dists_idx : array-like, shape=[...,]
        aligned_points : array-like, shape=[..., n_grid, n_nodes, n_nodes]
        """
        total_space = self._total_space
        geodesic_points = geodesic(self.s_grid)
        geo_shape = (self.n_grid,) + total_space.shape

        geo_batch_shape = get_batch_shape(3, geodesic_points)
        point_batch_shape = get_batch_shape(2, point)
        if point_batch_shape or geo_batch_shape:
            batch_shape = point_batch_shape or geo_batch_shape
        else:
            batch_shape = ()

        point = gs.broadcast_to(point, (self.n_grid,) + batch_shape + total_space.shape)
        if point_batch_shape:
            point = gs.moveaxis(point, 0, -3)

        if batch_shape and not geo_batch_shape:
            geodesic_points = gs.broadcast_to(geodesic_points, batch_shape + geo_shape)

        flat_point = gs.reshape(point, (-1,) + total_space.shape)
        flat_geodesic_s = gs.reshape(geodesic_points, (-1,) + total_space.shape)

        aligned_flat_points = total_space.aligner.align(flat_point, flat_geodesic_s)
        flat_dists = total_space.metric.dist(flat_geodesic_s, aligned_flat_points)

        perm_ = total_space.aligner.perm_
        total_space.aligner.align_algo.perm_ = gs.reshape(
            perm_, batch_shape + (self.n_grid, total_space.n_nodes)
        )

        dists = gs.reshape(
            flat_dists,
            batch_shape + (self.n_grid,),
        )

        min_dists_idx = gs.argmin(dists, axis=-1)

        aligned_points = gs.reshape(
            aligned_flat_points,
            batch_shape + geo_shape,
        )

        return dists, min_dists_idx, aligned_points

    def dist(self, geodesic, point):
        """Compute the distance between the geodesic and the point.

        Parameters
        ----------
        geodesic : function
            Geodesic function in GraphSpace.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        dist : array-like, shape=[..., n_nodes]
            Distance between the point and the geodesic.

        Notes
        -----
        Due to the discrete nature of the method, distance is not very accurate.
        """
        dists, min_dists_idx, _ = self._compute_dists(geodesic, point)
        slc = []
        for n in dists.shape[:-1]:
            slc.append(gs.arange(n))
        slc.append(min_dists_idx)
        return dists[tuple(slc)]

    def align(self, geodesic, point):
        """Align the graph to the geodesic.

        Parameters
        ----------
        geodesic : function
            Geodesic function in GraphSpace.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        aligned_point : array-like, shape=[..., n_nodes, n_nodes]
            Permuted graph as to be aligned with respect to the geodesic.
        """
        _, min_dists_idx, aligned_points = self._compute_dists(geodesic, point)
        slc = []
        for n in aligned_points.shape[:-3]:
            slc.append(gs.arange(n))
        slc.append(min_dists_idx)
        self.perm_ = self._total_space.aligner.perm_[tuple(slc)]

        slc.extend([slice(None), slice(None)])
        return aligned_points[tuple(slc)]


class _GeodesicToPointAligner(PointToGeodesicAlignerBase):
    """Class for the alignment of the points with respect to a geodesic.

    Solves a 1d optimization problem.

    Parameters
    ----------
    total_space : GraphSpace
        Set with quotient structure.
    save_opt_res : bool
        Whether to save optimization results.
    """

    def __init__(self, total_space, save_opt_res=False):
        super().__init__(total_space)

        self.save_opt_res = save_opt_res
        self.minimizer = ScipyMinimize(method="BFGS")

        self.opt_results_ = None

    def _objective_single(self, param, geodesic, point):
        """Objective function.

        Parameters
        ----------
        param : array-like, shape=[1,]
            Parameter along the geodesic.
        geodesic : function
            Geodesic function in GraphSpace.
        point : array-like, shape=[n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        dist : array-like, shape=[]
            Dist from point to geodesic.
        """
        geodesic_point = geodesic(param)
        if geodesic_point.ndim > 3:
            raise NotImplementedError("Cannot handle more than one geodesic at time")

        geodesic_point = gs.squeeze(geodesic_point, axis=0)
        return self._total_space.quotient.metric.squared_dist(geodesic_point, point)

    def _optimize_single(self, geodesic, point):
        """Solution of optimization problem.

        Parameters
        ----------
        geodesic : function
            Geodesic function in GraphSpace.
        point : array-like, shape=[n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        res : OptimizationResult
            Result of optimization.
        """

        def objective(param):
            return self._objective_single(param, geodesic=geodesic, point=point)

        return self.minimizer.minimize(
            objective,
            x0=0.0,
        )

    def squared_dist(self, geodesic, point, return_perm=False):
        """Compute the distance between the geodesic and the point.

        Parameters
        ----------
        geodesic : function
            Geodesic function in GraphSpace.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.
        return_perm : bool
            If to return optimal permutations.

        Returns
        -------
        sdist : array-like, shape=[...]
            Squared distance between point and geodesic.
        perm : array-like, shape=[..., n_nodes]
            Optimal permutations.
        """
        batch_shape = get_batch_shape(self._total_space.point_ndim, point)

        if not batch_shape:
            point = gs.expand_dims(point, axis=0)

        perms = []
        min_sdists = []
        opt_results = []
        for point_ in point:
            res = self._optimize_single(geodesic, point_)
            perms.append(self._total_space.aligner.perm_)
            min_sdists.append(res.fun)

            opt_results.append(res)

        if not batch_shape:
            min_sdists = min_sdists[0]
            perms = perms[0]
            opt_results = opt_results[0]

        if self.save_opt_res:
            self.opt_results_ = opt_results

        min_sdists = gs.array(min_sdists)
        if return_perm:
            return min_sdists, gs.array(perms)

        return min_sdists

    def dist(self, geodesic, point):
        """Compute the distance between the geodesic and the point.

        Parameters
        ----------
        geodesic : function
            Geodesic function in GraphSpace.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        dist : array-like, shape=[..., n_nodes]
            Distance between the point and the geodesic.
        """
        sdist, _ = self.squared_dist(geodesic, point, return_perm=True)
        return gs.sqrt(sdist)

    def align(self, geodesic, point):
        """Align the graph to the geodesic.

        Parameters
        ----------
        geodesic : function
            Geodesic function in GraphSpace.
        point : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        aligned_point : array-like, shape=[..., n_nodes, n_nodes]
            Permuted graph as to be aligned with respect to the geodesic.
        """
        _, self.perm_ = self.squared_dist(geodesic, point, return_perm=True)
        return self._total_space.group_action(self.perm_, point)


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

    def new(self, equip=True):
        """Create manifold with same parameters."""
        return GraphSpace(n_nodes=self.n_nodes, equip=equip)

    def equip_with_group_action(self, group_action="permutations"):
        """Equip manifold with group action."""
        if group_action == "permutations":
            group_action = PermutationAction()

        return super().equip_with_group_action(group_action)


class GraphSpaceAligner(Aligner):
    """Graph space aligner.

    Parameters
    ----------
    total_space : GraphSpace
        Set with quotient structure.
    align_algo : GraphSpaceAlignerAlgorithm
        Algorihtm performing alignment.
    """

    MAP_ALIGNER = {
        "FAQ": FAQAligner,
        "exhaustive": ExhaustiveAligner,
    }

    def __init__(self, total_space, align_algo=None):
        super().__init__(total_space=total_space, align_algo=align_algo)
        if align_algo is None:
            align_algo = self.set_alignment_algorithm()
        self.point_to_geodesic_aligner = self.set_point_to_geodesic_aligner()

    @property
    def perm_(self):
        """Optimal node permutations.

        Returns
        -------
        perm_ : array-like, shape=[..., n_nodes]
            Node permutations where in position i we have the value j meaning
        the node i should be permuted with node j.
        """
        return self.align_algo.perm_

    def set_alignment_algorithm(self, align_algo="FAQ", **kwargs):
        """Set the aligning strategy.

        GraphSpace metric relies on alignment. In this module we propose the
        the FAQ graph matching by [Vogelstein2015]_, and
        exhaustive aligner which explores the whole permutation group.

        Parameters
        ----------
        align_algo : str or GraphSpaceAlignerAlgorithm
            'FAQ': Fast Quadratic Assignment - only compatible with Frobenius norm,
            'exhaustive': all group exhaustive search
        """
        if isinstance(align_algo, str):
            check_parameter_accepted_values(
                align_algo, "align_algo", list(self.MAP_ALIGNER.keys())
            )

            aligner_algorithm = self.MAP_ALIGNER.get(align_algo)(
                self._total_space, **kwargs
            )

        self.align_algo = aligner_algorithm
        return self.align_algo

    def set_point_to_geodesic_aligner(self, aligner="default", **kwargs):
        """Set the alignment between a point and a geodesic.

        Following the geodesic to point alignment in [Calissano2020]_ and
        [Huckemann2010]_, this function defines the parameters [s_min, s_max] and
        the number of points to sample in the domain.

        Parameters
        ----------
        aligner: BasePointToGeodesicAligner
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
            kwargs.setdefault("n_grid", 10)
            aligner = PointToGeodesicAligner(self._total_space, **kwargs)

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
        return self.point_to_geodesic_aligner.align(geodesic, point)


class GraphSpaceQuotientMetric(QuotientMetric):
    r"""Class for the Graph Space Metric.

    Every metric :math:`d: X \times X \rightarrow \mathbb{R}` on the total space of
    adjacency matrices can descend onto the quotient space as a pseudo-metric:
    :math:`d([x_1],[x_2]) = min_{t\in T} d_X(x_1, t^Tx_2t)`. The metric relies on the
    total space metric and an alignment procedure, i.e., Graph Matching or Networks
    alignment procedure. Metric, alignment, geodesics, and alignment with respect to
    a geodesic are defined. By default, the alignment is FAQ and the total
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


register_quotient(
    Space=GraphSpace,
    Metric=MatricesMetric,
    GroupAction=PermutationAction,
    FiberBundle=GraphSpaceAligner,
    QuotientMetric=GraphSpaceQuotientMetric,
)
