"""Classes for the pullback metric.

Lead author: Nina Miolane.
"""

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.numerics.geodesic import ExpODESolver, LogShootingSolver
from geomstats.numerics.ivp import GSIVPIntegrator
from geomstats.vectorization import check_is_batch


class PullbackMetric(RiemannianMetric):
    r"""Pullback metric.

    Let :math:`f` be an immersion :math:`f: M \rightarrow N`
    of one manifold :math:`M` into the Riemannian manifold :math:`N`
    with metric :math:`g`.
    The pull-back metric :math:`f^*g` is defined on :math:`M` for a
    base point :math:`p` as:
    :math:`(f^*g)_p(u, v) = g_{f(p)}(df_p u , df_p v)
    \quad \forall u, v \in T_pM`

    Note
    ----
    The pull-back metric is currently only implemented for an
    immersion into the Euclidean space, i.e. for

    :math:`N=\mathbb{R}^n`.
    """

    def __init__(self, space, signature=None):
        super().__init__(space, signature)
        self._instantiate_solvers()

    def _instantiate_solvers(self):
        if not gs.__name__.endswith("numpy"):
            self.log_solver = LogShootingSolver()

        self.exp_solver = ExpODESolver(
            integrator=GSIVPIntegrator(n_steps=100, step_type="euler"),
        )

    def metric_matrix(self, base_point):
        r"""Metric matrix at the tangent space at a base point.

        Let :math:`f` be the immersion
        :math:`f: M \rightarrow \mathbb{R}^n` of the manifold
        :math:`M` into the Euclidean space :math:`\mathbb{R}^n`.
        The elements of the metric matrix at a base point :math:`p`
        are defined as:
        :math:`(f*g)_{ij}(p) = <df_p e_i , df_p e_j>`,
        for :math:`e_i, e_j` basis elements of :math:`M`.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        dim, dim_embedding = self._space.dim, self._space.embedding_space.dim
        is_vec = check_is_batch(self._space.point_ndim, base_point)

        immersed_base_point = self._space.immersion(base_point)
        jacobian_immersion = self._space.jacobian_immersion(base_point)

        basis_elements = gs.eye(dim)

        if is_vec:
            reshaped_jacobian_immersion = gs.reshape(jacobian_immersion, (-1, dim))
            reshaped_immersed_basis_elements = gs.matvec(
                reshaped_jacobian_immersion, basis_elements
            )
            immersed_basis_elements = gs.moveaxis(
                gs.reshape(reshaped_immersed_basis_elements, (dim, -1, dim_embedding)),
                0,
                1,
            )

        else:
            immersed_basis_elements = gs.matvec(jacobian_immersion, basis_elements)

        elems = {}
        for i in range(dim):
            for j in range(i, dim):
                elem = self._space.embedding_space.metric.inner_product(
                    immersed_basis_elements[..., i, :],
                    immersed_basis_elements[..., j, :],
                    immersed_base_point,
                )
                elems[(i, j)] = elem

        mat = []
        for i in range(dim):
            for j in range(dim):
                elem = elems[(i, j)] if j > i else elems[(j, i)]
                mat.append(elem)

        shape = (-1, dim, dim) if is_vec else (dim, dim)
        return gs.reshape(gs.stack(mat, axis=-1), shape)

    def inner_product_derivative_matrix(self, base_point):
        r"""Compute the inner-product derivative matrix.

        The derivative of the metrix matrix is given by
        :math:`\partial_k g_{ij}(p)`
        where :math:`p` is the base_point.

        The index k of the derivation is last.

        Parameters
        ----------
        base_point : array-like, shape=[..., *shape]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_prod_deriv_mat : array-like, shape=[..., dim, dim, dim]
            Inner-product derivative matrix, where the index of the derivation
            is last: :math:`mat_{ijk} = \partial_k g_{ij}`.
        """
        jacobian_ai = self._space.jacobian_immersion(base_point)
        hessian_aij = self._space.hessian_immersion(base_point)
        return gs.einsum("...aki,...aj->...ijk", hessian_aij, jacobian_ai) + gs.einsum(
            "...akj,...ai->...ijk", hessian_aij, jacobian_ai
        )

    def second_fundamental_form(self, base_point):
        r"""Compute the second fundamental form.

        In the case of an immersion f, the second fundamental form is
        given by the formula:
        :math:`\RN{2}(p)_{ij}^\alpha = \partial_{i j}^2 f^\alpha(p)`
        :math:`-\Gamma_{i j}^k(p) \partial_k f^\alpha(p)`
        at base_point :math:`p`.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        second_fundamental_form : array-like, shape=[..., embedding_dim, dim, dim]
            Second fundamental form :math:`\RN{2}(p)_{ij}^\alpha` where the
            :math:`\alpha` index is first.
        """
        christoffels = self.christoffels(base_point)

        jacobian = self._space.jacobian_immersion(base_point)
        hessian = self._space.hessian_immersion(base_point)

        return hessian - gs.einsum("...kij,...dk->...dij", christoffels, jacobian)

    def mean_curvature_vector(self, base_point):
        r"""Compute the mean curvature vector.

        The mean curvature vector is defined at base point :math:`p` by
        :math:`H_p^\alpha= \frac{1}{d} (f^{*}g)_{p}^{ij} (\partial_{i j}^2 f^\alpha(p)`
        :math:`-\Gamma_{i j}^k(p) \partial_k f^\alpha(p))`
        where :math:`f^{*}g` is the pullback of the metric :math:`g` by the
        immersion :math:`f`.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        mean_curvature_vector : array-like, shape=[..., embedding_dim]
            Mean curvature vector.
        """
        second_fund_form = self.second_fundamental_form(base_point)
        cometric = self.cometric_matrix(base_point)
        return gs.einsum("...ij,...aij->...a", cometric, second_fund_form)


class PullbackDiffeoMetric(RiemannianMetric):
    """Pullback metric induced by a diffeomorphism."""

    def __init__(self, space, diffeo, image_space, signature=None):
        super().__init__(space, signature=signature)
        self.diffeo = diffeo
        self.image_space = image_space

    def metric_matrix(self, base_point=None):
        """Metric matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., *shape]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        raise NotImplementedError(
            "The computation of the pullback metric matrix"
            " is not implemented yet in general shape setting."
        )

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., *shape]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., *shape]
            Tangent vector at base point.
        base_point: array-like, shape=[..., *shape]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        image_point = self.diffeo.diffeomorphism(base_point)
        return self.image_space.metric.inner_product(
            self.diffeo.tangent_diffeomorphism(
                tangent_vec_a,
                base_point=base_point,
                image_point=image_point,
            ),
            self.diffeo.tangent_diffeomorphism(
                tangent_vec_b,
                base_point=base_point,
                image_point=image_point,
            ),
            image_point,
        )

    def squared_norm(self, vector, base_point):
        """Compute the square of the norm of a vector.

        Squared norm of a vector associated to the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm.
        """
        image_point = self.diffeo.diffeomorphism(base_point)
        return self.image_space.metric.squared_norm(
            self.diffeo.tangent_diffeomorphism(
                vector,
                base_point=base_point,
                image_point=image_point,
            ),
            image_point,
        )

    def norm(self, vector, base_point):
        """Compute norm of a vector.

        Norm of a vector associated to the inner product
        at the tangent space at a base point.

        Note: This only works for positive-definite
        Riemannian metrics and inner products.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        norm : array-like, shape=[...,]
            Norm.
        """
        image_point = self.diffeo.diffeomorphism(base_point)
        return self.image_space.metric.norm(
            self.diffeo.tangent_diffeomorphism(
                vector,
                base_point=base_point,
                image_point=image_point,
            ),
            image_point,
        )

    def exp(self, tangent_vec, base_point):
        """Compute the exponential map via diffeomorphic pullback.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *shape]
            Tangent vector.
        base_point : array-like, shape=[..., *shape]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., *shape]
            Riemannian exponential of tangent_vec at base_point.
        """
        image_base_point = self.diffeo.diffeomorphism(base_point)
        image_tangent_vec = self.diffeo.tangent_diffeomorphism(
            tangent_vec, base_point=base_point, image_point=image_base_point
        )
        image_exp = self.image_space.metric.exp(
            image_tangent_vec,
            image_base_point,
        )
        return self.diffeo.inverse_diffeomorphism(image_exp)

    def log(self, point, base_point):
        """Compute the logarithm map via diffeomorphic pullback.

        Parameters
        ----------
        point : array-like, shape=[..., *shape]
            Point.
        base_point : array-like, shape=[..., *shape]
            Point.

        Returns
        -------
        log : array-like, shape=[..., *shape]
            Logarithm of point from base_point.
        """
        image_base_point = self.diffeo.diffeomorphism(base_point)
        image_point = self.diffeo.diffeomorphism(point)
        image_log = self.image_space.metric.log(image_point, image_base_point)
        return self.diffeo.inverse_tangent_diffeomorphism(
            image_log, image_point=image_base_point, base_point=base_point
        )

    def squared_dist(self, point_a, point_b):
        """Squared geodesic distance between two points via pullback.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point.
        point_b : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
            Squared distance.
        """
        image_point_a = self.diffeo.diffeomorphism(point_a)
        image_point_b = self.diffeo.diffeomorphism(point_b)
        return self.image_space.metric.squared_dist(image_point_a, image_point_b)

    def dist(self, point_a, point_b):
        """Compute the distance via diffeomorphic pullback.

        Parameters
        ----------
        point_a : array-like, shape=[..., *shape]
            Point a.
        point_b : array-like, shape=[..., *shape]
            Point b.

        Returns
        -------
        dist : array-like
            Distance between point_a and point_b.
        """
        image_point_a = self.diffeo.diffeomorphism(point_a)
        image_point_b = self.diffeo.diffeomorphism(point_b)
        return self.image_space.metric.dist(image_point_a, image_point_b)

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        """Compute the geodesic via diffeomorphic pullback.

        Parameters
        ----------
        initial_point : array-like, shape=[..., *shape]
            Initial Point.
        end_point : array-like, shape=[..., *shape]
            End Point.

        Returns
        -------
        geodesic : callable
            Geodesic between initial_point and end_point.
        """
        image_initial_point = self.diffeo.diffeomorphism(initial_point)
        if end_point is None:
            image_end_point = None
        else:
            image_end_point = self.diffeo.diffeomorphism(end_point)
        if initial_tangent_vec is None:
            image_initial_tangent_vec = None
        else:
            image_initial_tangent_vec = self.diffeo.tangent_diffeomorphism(
                initial_tangent_vec,
                base_point=initial_point,
                image_point=image_initial_point,
            )
        image_geodesic = self.image_space.metric.geodesic(
            image_initial_point, image_end_point, image_initial_tangent_vec
        )

        def path(t):
            image_point_t = image_geodesic(t)
            point_t = self.diffeo.inverse_diffeomorphism(image_point_t)
            return point_t

        return path

    def curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point):
        """Compute the curvature via diffeomorphic pullback.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *shape]
            Tangent vector a.
        tangent_vec_b : array-like, shape=[..., *shape]
            Tangent vector b.
        tangent_vec_c : array-like, shape=[..., *shape]
            Tangent vector c.
        base_point : array-like, shape=[..., *shape]
            Base point.

        Returns
        -------
        curvature : array-like, shape=[..., *shape]
            Curvature in directions tangent_vec a, b, c at base_point.
        """
        image_base_point = self.diffeo.diffeomorphism(base_point)
        image_tangent_vec_a = self.diffeo.tangent_diffeomorphism(
            tangent_vec_a,
            base_point=base_point,
            image_point=image_base_point,
        )
        image_tangent_vec_b = self.diffeo.tangent_diffeomorphism(
            tangent_vec_b,
            base_point=base_point,
            image_point=image_base_point,
        )
        image_tangent_vec_c = self.diffeo.tangent_diffeomorphism(
            tangent_vec_c,
            base_point=base_point,
            image_point=image_base_point,
        )
        image_curvature = self.image_space.metric.curvature(
            image_tangent_vec_a,
            image_tangent_vec_b,
            image_tangent_vec_c,
            image_base_point,
        )
        return self.diffeo.inverse_tangent_diffeomorphism(
            image_curvature, image_point=image_base_point, base_point=base_point
        )

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        """Compute the parallel transport via diffeomorphic pullback.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *shape]
            Tangent vector.
        base_point : array-like, shape=[..., *shape]
            Base point.
        direction : array-like, shape=[..., *shape]
            Direction.
        end_point : array-like, shape=[..., *shape]
            End point.

        Returns
        -------
        parallel_transport : array-like, shape=[..., *shape]
            Parallel transport.
        """
        image_base_point = self.diffeo.diffeomorphism(base_point)
        image_tangent_vec = self.diffeo.tangent_diffeomorphism(
            tangent_vec, base_point=base_point, image_point=image_base_point
        )

        image_direction = (
            None
            if direction is None
            else self.diffeo.tangent_diffeomorphism(
                direction, base_point=base_point, image_point=image_base_point
            )
        )
        image_end_point = (
            None if end_point is None else self.diffeo.diffeomorphism(end_point)
        )

        image_parallel_transport = self.image_space.metric.parallel_transport(
            image_tangent_vec,
            image_base_point,
            direction=image_direction,
            end_point=image_end_point,
        )

        if image_end_point is None:
            image_end_point = self.image_space.metric.exp(
                image_direction, image_base_point
            )

        return self.diffeo.inverse_tangent_diffeomorphism(
            image_parallel_transport, image_point=image_end_point, base_point=end_point
        )
