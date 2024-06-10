"""Class for (principal) fiber bundles.

Lead author: Nicolas Guigui.
"""

import logging
import math
from abc import ABC, abstractmethod

import geomstats.backend as gs
from geomstats.numerics.optimizers import ScipyMinimize
from geomstats.vectorization import check_is_batch, get_batch_shape


def _from_base(method):
    """Decorate method in order to avoid recursive calls."""
    method._from_base = True
    return method


class AlignerAlgorithm(ABC):
    """Base class for point to point aligner.

    Parameters
    ----------
    total_space : Manifold
        Space equipped with a group action and a group-invariant metric.
    """

    def __init__(self, total_space):
        self._total_space = total_space

    @abstractmethod
    def align(self, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        point : array-like, shape=[..., *total_space.shape]
            Point to align.
        base_point : array-like, shape=[..., *total_space.shape]
            Base point.

        Returns
        -------
        aligned_point : array-like, shape=[..., *total_space.shape]
            Aligned point.
        """


class DistanceMinimizingAligner(AlignerAlgorithm):
    """Aligment based on minimization of squared distance.

    Parameters
    ----------
    total_space : Manifold
        Space equipped with a group action and a group-invariant metric.
    optimizer : ScipyMinimize
        Optimizer to solve minimization problem.
    group_elem_shape : tuple
        Shape of the group element representation.
    """

    def __init__(self, total_space, optimizer=None, group_elem_shape=None):
        super().__init__(total_space)
        if optimizer is None:
            optimizer = ScipyMinimize(
                method="L-BFGS-B",
                jac="autodiff",
            )
        self.optimizer = optimizer

        if group_elem_shape is None:
            group_elem_shape = self._total_space.group_action.group_elem_shape
        self.group_elem_shape = group_elem_shape

    def _objective(self, point, base_point, batch_shape):
        """Objective function.

        Parameters
        ----------
        point : array-like, shape=[..., *total_space.shape]
            Point to align to base point.
        base_point : array-like, shape=[..., *total_space.shape]
            Point wrt alignment is performed.
        batch_shape : tuple
            Batch shape.
        """

        def sum_squared_dist(param):
            """Objective function.

            Parameters
            ----------
            param : array-like
                Flat representation of group element.
            """
            group_elem = gs.reshape(param, batch_shape + self.group_elem_shape)
            aligned_point = self._total_space.group_action(group_elem, point)
            return gs.sum(
                self._total_space.metric.squared_dist(aligned_point, base_point)
            )

        return sum_squared_dist

    def align(self, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        point : array-like, shape=[..., *total_space.shape]
            Point to align.
        base_point : array-like, shape=[..., *total_space.shape]
            Base point.

        Returns
        -------
        aligned_point : array-like, shape=[..., *total_space.shape]
            Aligned point.
        """
        batch_shape = get_batch_shape(self._total_space.point_ndim, point, base_point)
        objective = self._objective(point, base_point, batch_shape)

        initial_param = gs.flatten(
            gs.random.rand(math.prod(batch_shape + self.group_elem_shape))
        )

        sol = self.optimizer.minimize(
            objective,
            initial_param,
        )

        group_elem = gs.reshape(sol.x, batch_shape + self.group_elem_shape)
        aligned_point = self._total_space.group_action(group_elem, point)

        return aligned_point


class AlternatingAligner(AlignerAlgorithm):
    """Alternate alignment algorithm.

    Assumes total space is equipped with several group actions.
    Aligns points wrt these group actions by alternate minimization
    wrt each of them (similar approach used in e.g. [JKKS2012]_).

    Parameters
    ----------
    total_space : Manifold
        Manifold equipped with a quotient structure.
    threshold : float
        Distance between consecutive aligned points for which
        convergence is considered reached.
    max_iter : int
        Maximum number of iterations.
    verbose : boolean
        If log number of iterations need for convergence.

    References
    ----------
    .. [JKKS2012] Ian H. Jermyn, Sebastian Kurtek, Eric Klassen, and Anuj Srivastava.
        “Elastic Shape Matching of Parameterized Surfaces Using Square Root Normal
        Fields.” In Computer Vision – ECCV 2012, edited by Andrew Fitzgibbon,
        Svetlana Lazebnik, Pietro Perona, Yoichi Sato, and Cordelia Schmid,
        804–17. Lecture Notes in Computer Science. Berlin, Heidelberg: Springer, 2012.
        https://doi.org/10.1007/978-3-642-33715-4_58.
    """

    def __init__(self, total_space, threshold=1e-3, max_iter=20, verbose=0):
        super().__init__(total_space=total_space)

        self.threshold = threshold
        self.max_iter = max_iter
        self.verbose = verbose

        self.total_spaces = []
        for group_action in total_space.group_action:
            total_space = total_space.new(equip=True)
            total_space.equip_with_group_action(group_action)
            total_space.equip_with_quotient()
            self.total_spaces.append(total_space)

    def _align_single(self, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        point : array-like, shape=[*point_shape]
            Discrete curve to align.
        base_point : array-like, shape=[*point_shape]
            Reference discrete curve.

        Returns
        -------
        aligned : array-like, shape=[*point_shape]
            Aligned point.
        """
        aligned_point = previous_aligned_point = point
        for index in range(self.max_iter):
            for total_space in self.total_spaces:
                aligned_point = total_space.fiber_bundle.align(
                    aligned_point, base_point
                )

            gap = self._total_space.metric.dist(aligned_point, previous_aligned_point)
            previous_aligned_point = aligned_point

            if gap < self.threshold:
                if self.verbose > 0:
                    logging.info(
                        f"Convergence of alignment reached after {index + 1} "
                        "iterations."
                    )

                break
        else:
            logging.warning(
                f"Maximum number of iterations {self.max_iter} reached during "
                "alignment. The result may be inaccurate."
            )
        return aligned_point

    def align(self, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Discrete curve to align.
        base_point : array-like, shape=[..., *point_shape]
            Reference discrete curve.

        Returns
        -------
        aligned : array-like, shape=[..., *point_shape]
            Aligned point.
        """
        is_batch = check_is_batch(
            self._total_space.point_ndim,
            point,
            base_point,
        )
        if not is_batch:
            return self._align_single(point, base_point)

        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        return gs.stack(
            [
                self._align_single(point_, base_point_)
                for point_, base_point_ in zip(point, base_point)
            ]
        )


class FiberBundle:
    """Class for (principal) fiber bundles.

    This class implements abstract methods for fiber bundles, or more
    generally manifolds, with a submersion map, or a right Lie group action.

    Parameters
    ----------
    total_space : Manifold
        Space equipped with a group action and a group-invariant metric.
    aligner : AlignerAlgorithm
        If True and autodiff works, instantiates default
        DistanceMinimizationBasedAligner.
    """

    def __init__(self, total_space, aligner=None):
        self._total_space = total_space
        if aligner is True:
            if isinstance(total_space.group_action, tuple):
                aligner = AlternatingAligner(total_space)
            else:
                aligner = (
                    DistanceMinimizingAligner(total_space)
                    if not gs.__name__.endswith("numpy")
                    else None
                )

        self.aligner = aligner

    @staticmethod
    def riemannian_submersion(point):
        """Project a point to base manifold.

        This is the projection of the fiber bundle, defined on the total
        space, with values in the base manifold. This map is surjective.
        By default, the base manifold is not explicit but is identified with a
        local section of the fiber bundle, so the submersion is the identity
        map.

        Parameters
        ----------
        point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point of the total space.

        Returns
        -------
        projection : array-like, shape=[..., {base_dim, [n, m]}]
            Point of the base manifold.
        """
        return gs.copy(point)

    @staticmethod
    def lift(point):
        """Lift a point to total space.

        This is a section of the fiber bundle, defined on the base manifold,
        with values in the total space. It means that submersion applied after
        lift results in the identity map. By default, the base manifold
        is not explicit but is identified with a section of the fiber bundle,
        so the lift is the identity map.

        Parameters
        ----------
        point : array-like, shape=[..., {base_dim, [n, m]}]
            Point of the base manifold.

        Returns
        -------
        lift : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point of the total space.
        """
        return gs.copy(point)

    @_from_base
    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        """Project a tangent vector to base manifold.

        This is the differential of the projection of the fiber bundle,
        defined on the tangent space of a point of the total space,
        with values in the tangent space of the projection of this point in the
        base manifold. This map is surjective. By default, the base manifold
        is not explicit but is identified with a horizontal section of the
        fiber bundle, so the tangent submersion is the horizontal projection.

        Parameters
        ----------
        tangent_vec :  array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector to the total space at `base_point`.
        base_point: array-like, shape=[..., {total_space.dim, [n, m]}]
            Point of the total space.

        Returns
        -------
        projection: array-like, shape=[..., {base_dim, [n, m]}]
            Tangent vector to the base manifold.
        """
        return self.horizontal_projection(tangent_vec, base_point)

    def align(self, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        point : array-like, shape=[..., *total_space.shape]
            Point to align.
        base_point : array-like, shape=[..., *total_space.shape]
            Base point.

        Returns
        -------
        aligned_point : array-like, shape=[..., *total_space.shape]
            Aligned point.
        """
        if self.aligner is None:
            raise NotImplementedError("Alignment is not implemented.")
        return self.aligner.align(point, base_point)

    @_from_base
    def horizontal_projection(self, tangent_vec, base_point):
        r"""Project to horizontal subspace.

        Compute the horizontal component of a tangent vector at a
        base point from:
            1. the vertical projection
            2. the horizontal lift of the tangent submersion

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point on the total space.

        Returns
        -------
        horizontal : array-like, shape=[..., {total_space.dim, [n, m]}]
            Horizontal component of `tangent_vec`.
        """
        if not hasattr(self.vertical_projection, "_from_base"):
            ver_tangent_vec = self.vertical_projection(tangent_vec, base_point)
            return tangent_vec - ver_tangent_vec

        if not (
            hasattr(self.horizontal_lift, "_from_base")
            or hasattr(self.tangent_riemannian_submersion, "_from_base")
        ):
            return self.horizontal_lift(
                self.tangent_riemannian_submersion(tangent_vec, base_point),
                fiber_point=base_point,
            )

        raise NotImplementedError("Horizontal projection is not implemented.")

    @_from_base
    def vertical_projection(self, tangent_vec, base_point):
        r"""Project to vertical subspace.

        Compute the vertical component of a tangent vector :math:`w` at a
        base point :math:`P` by removing the horizontal component.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point on the total space.

        Returns
        -------
        vertical : array-like, shape=[..., {total_space.dim, [n, m]}]
            Vertical component of `tangent_vec`.
        """
        return tangent_vec - self.horizontal_projection(tangent_vec, base_point)

    def is_horizontal(self, tangent_vec, base_point, atol=gs.atol):
        """Evaluate if the tangent vector is horizontal at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point on the manifold.
            Optional, default: None.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol

        Returns
        -------
        is_horizontal : bool
            Boolean denoting if tangent vector is horizontal.
        """
        return gs.all(
            gs.isclose(
                tangent_vec,
                self.horizontal_projection(tangent_vec, base_point),
                atol=atol,
            ),
            axis=(-2, -1),
        )

    def is_vertical(self, tangent_vec, base_point, atol=gs.atol):
        """Evaluate if the tangent vector is vertical at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point on the manifold.
            Optional, default: None.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_vertical : bool
            Boolean denoting if tangent vector is vertical.
        """
        return gs.all(
            gs.isclose(
                0.0,
                self.tangent_riemannian_submersion(tangent_vec, base_point),
                atol=atol,
            ),
            axis=(-2, -1),
        )

    @_from_base
    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        """Lift a tangent vector to a horizontal vector in the total space.

        It means that horizontal lift is the inverse of the restriction of the
        tangent submersion to the horizontal space at point, where point must
        be in the fiber above the base point. By default, the base manifold
        is not explicit but is identified with a horizontal section of the
        fiber bundle, so the horizontal lift is the horizontal projection.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {base_dim, [n, m]}]
        fiber_point : array-like, shape=[..., {ambient_dim, [n, m]}]
            Point of the total space.
            Optional, default : None. The `lift` method is used to compute a
            point at which to compute a tangent vector.
        base_point : array-like, shape=[..., {base_dim, [n, m]}]
            Point of the base space.
            Optional, default : None. In this case, point must be given,
            and `submersion` is used to compute the base_point if needed.

        Returns
        -------
        horizontal_lift : array-like, shape=[..., {total_space.dim, [n, m]}]
            Horizontal tangent vector to the total space at point.
        """
        if base_point is None and fiber_point is None:
            raise ValueError(
                "Either a point (of the total space) or a "
                "base point (of the base manifold) must be "
                "given."
            )

        if fiber_point is None:
            fiber_point = self.lift(base_point)

        return self.horizontal_projection(tangent_vec, fiber_point)

    def integrability_tensor(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the fundamental tensor A of the submersion.

        The fundamental integrability tensor A is defined for tangent vectors
        :math:`X = tangent\_vec\_a` and :math:`Y = tangent\_vec\_b` of the
        total space by [ONeill]_ as
        :math:`A_X Y = ver\nabla_{hor X} (hor Y) + hor \nabla_{hor X}( ver Y)`
        where :math:`hor, ver` are the horizontal and vertical projections
        and :math:`\nabla` is the connection of the total space.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point of the total space.

        Returns
        -------
        vector : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`, result of the A tensor applied to
            `tangent_vec_a` and `tangent_vec_b`.

        References
        ----------
        .. [ONeill]  O’Neill, Barrett. The Fundamental Equations of a
            Submersion, Michigan Mathematical Journal 13, no. 4
            (December 1966): 459–69. https://doi.org/10.1307/mmj/1028999604.
        """
        raise NotImplementedError

    def integrability_tensor_derivative(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        nabla_x_y,
        tangent_vec_e,
        nabla_x_e,
        base_point,
    ):
        r"""Compute the covariant derivative of the integrability tensor A.

        The covariant derivative :math:`\nabla_X (A_Y E)` in total space is
        necessary to compute the covariant derivative of the directional
        curvature in a submersion. The components :math:`\nabla_X (A_Y E)`
        and :math:`A_Y E` are computed at base-point :math:`P = base\_point`
        for horizontal vector fields :math:`X, Y` extending the values
        given in argument :math:`X|_P = horizontal\_vec\_x`,
        :math:`Y|_P = horizontal\_vec\_y` and a general vector field
        :math:`E` extending :math:`E|_x = tangent\_vec\_e`
        in a neighborhood of x with covariant derivatives
        :math:`\nabla_X Y |_P = nabla_x y` and
        :math:`\nabla_X E |_P = nabla_x e`.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., {total_space.dim, [n, m]}]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., {total_space.dim, [n, m]}]
            Horizontal tangent vector at `base_point`.
        nabla_x_y : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`.
        tangent_vec_e : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`.
        nabla_x_e : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point of the total space.

        Returns
        -------
        nabla_x_a_y_e : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`, result of :math:`\nabla_X
            (A_Y E)`.
        a_y_e : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector at `base_point`, result of :math:`A_Y E`.

        References
        ----------
        .. [Pennec] Pennec, Xavier. Computing the curvature and its gradient
            in Kendall shape spaces. Unpublished.
        """
        raise NotImplementedError
