"""The vector space of symmetric matrices.

Lead author: Yann Thanwerdas.
"""

import geomstats.backend as gs
from geomstats.geometry.base import (
    DiffeomorphicMatrixVectorSpace,
    LevelSet,
    MatrixVectorSpace,
)
from geomstats.geometry.diffeo import Diffeo
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.vectorization import repeat_out, repeat_out_multiple_ndim


class SymmetricMatrices(MatrixVectorSpace):
    """Class for the vector space of symmetric matrices of size n.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.
    """

    def __init__(self, n, equip=True):
        self.n = n
        super().__init__(dim=int(n * (n + 1) / 2), shape=(n, n), equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MatricesMetric

    def _create_basis(self):
        """Compute the basis of the vector space of symmetric matrices."""
        indices, values = [], []
        k = -1
        for row in range(self.n):
            for col in range(row, self.n):
                k += 1
                if row == col:
                    indices.append((k, row, row))
                    values.append(1.0)
                else:
                    indices.extend([(k, row, col), (k, col, row)])
                    values.extend([1.0, 1.0])

        return gs.array_from_sparse(indices, values, (k + 1, self.n, self.n))

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a matrix is symmetric.

        Parameters
        ----------
        point : array-like, shape=[.., n, n]
            Point to test.
        atol : float
            Tolerance to evaluate equality with the transpose.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        belongs = super().belongs(point)
        if gs.any(belongs):
            is_symmetric = Matrices.is_symmetric(point, atol)
            return gs.logical_and(belongs, is_symmetric)
        return belongs

    def projection(self, point):
        """Make a matrix symmetric, by averaging with its transpose.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        sym : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        return Matrices.to_symmetric(point)

    @staticmethod
    def basis_representation(matrix_representation):
        """Convert a symmetric matrix into a vector.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., n(n+1)/2]
            Vector.
        """
        return gs.triu_to_vec(matrix_representation)

    @staticmethod
    def matrix_representation(basis_representation):
        """Convert a vector into a symmetric matrix.

        Parameters
        ----------
        basis_representation : array-like, shape=[..., n(n+1)/2]
            Vector.

        Returns
        -------
        matrix_representation : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        vec_dim = basis_representation.shape[-1]
        mat_dim = (gs.sqrt(8.0 * vec_dim + 1) - 1) / 2
        if mat_dim != int(mat_dim):
            raise ValueError(
                "Invalid input dimension, it must be of the form"
                "(n_samples, n * (n + 1) / 2)"
            )
        mat_dim = int(mat_dim)
        shape = (mat_dim, mat_dim)
        mask = 2 * gs.ones(shape) - gs.eye(mat_dim)
        indices = list(zip(*gs.triu_indices(mat_dim)))
        if gs.ndim(basis_representation) == 1:
            upper_triangular = gs.array_from_sparse(
                indices, basis_representation, shape
            )
        else:
            upper_triangular = gs.stack(
                [
                    gs.array_from_sparse(indices, data, shape)
                    for data in basis_representation
                ]
            )

        mat = Matrices.to_symmetric(upper_triangular) * mask
        return mat


class SymmetricHollowMatrices(LevelSet, MatrixVectorSpace):
    r"""Space of symmetric hollow matrices.

    Set of symmetric matrices with null diagonal:

    .. math::

        \operatorname{Hol}(n) = \{X \in \operatorname{Sym}(n)
        \mid \operatorname{Diag}(X)=0\}

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.

    References
    ----------
    .. [T2022] Yann Thanwerdas. Riemannian and stratified
        geometries on covariance and correlation matrices. Differential
        Geometry [math.DG]. Université Côte d'Azur, 2022.
    """

    def __init__(self, n, equip=True):
        self.n = n
        super().__init__(dim=int(n * (n - 1) / 2), shape=(n, n), equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MatricesMetric

    def _define_embedding_space(self):
        return SymmetricMatrices(n=self.n)

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]

        Returns
        -------
        submersed_point : array-like, shape=[..., n]
        """
        return Matrices.diagonal(point)

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
        point : Ignored.

        Returns
        -------
        submersed_vector : array-like, shape=[..., n]
        """
        out = self.submersion(vector)
        return repeat_out(self.point_ndim, out, vector, point, out_shape=(self.n,))

    def _create_basis(self):
        """Compute the basis of the vector space of hollow symmetric matrices."""
        indices, values = [], []
        k = -1
        for row in range(self.n):
            for col in range(row + 1, self.n):
                k += 1
                indices.extend([(k, row, col), (k, col, row)])
                values.extend([1.0, 1.0])

        return gs.array_from_sparse(indices, values, (k + 1, self.n, self.n))

    @staticmethod
    def basis_representation(matrix_representation):
        """Convert a hollow symmetric matrix into a vector.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        vec : array-like, shape=[..., n(n+1)/2]
            Vector.
        """
        return gs.triu_to_vec(matrix_representation, k=1)

    def projection(self, point):
        """Project a point in embedding manifold on embedded manifold.

        Parameters
        ----------
        point : array-like, shape=[..., *embedding_space.point_shape]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., *point_shape]
            Projected point.
        """
        return point - Matrices.to_diagonal(point)


class HollowMatricesPermutationInvariantMetric(EuclideanMetric):
    r"""A permutation-invariant metric on the space of hollow matrices.

    It is flat Riemannian metric invariant by the congruence action
    of permutation matrices defined over a matrix vector space.

    Its associated quadratic form is:

    .. math::

        q(X)=\alpha \operatorname{tr}\left(X^2\right)
        +\beta \operatorname{Sum}\left(X^2\right)
        +\gamma \operatorname{Sum}(X)^2

    Parameters
    ----------
    space : Manifold
    alpha : float
        Scalar multiplying first term of quadratic form.
    beta : float
        Scalar multiplying second term of quadratic form.
    gamma : float
        Scalar multiplying third term of quadratic form.

    Check out chapter 8 of [T2022]_ for more details.

    References
    ----------
    .. [T2022] Yann Thanwerdas. Riemannian and stratified
        geometries on covariance and correlation matrices. Differential
        Geometry [math.DG]. Université Côte d'Azur, 2022.
    """

    def __init__(self, space, alpha=1.0, beta=1.0, gamma=1.0):
        self._check_params(space, alpha, beta, gamma)
        super().__init__(space=space)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _check_params(space, alpha, beta, gamma):
        r"""Check parameters of quadratic form.

        The following conditions must verify:
        - n > 3: :math:`\alpha>0,2 \alpha+(n-2) \beta>0, \alpha+(n-1)(\beta+n \gamma)>0`
        - n = 3: :math:`\alpha=0, \beta > 0, \beta+3 \gamma>0`
        - n = 2: :math:`\alpha=0, \beta=0, \gamma > 0`
        """
        n = space.n
        if n == 2:
            if alpha > gs.atol or beta > gs.atol or gamma < gs.atol:
                raise ValueError(
                    f"When n==2: alpha ({alpha}) and beta({beta}) must be 0,"
                    f"and gamma ({gamma}) > 0. "
                )
            return

        elif n == 3:
            cond = beta + 3 * gamma
            if alpha > gs.atol or beta < gs.atol or cond < gs.atol:
                raise ValueError(
                    f"When n==3: alpha ({alpha}) must be 0, beta ({beta}) > 0"
                    f"and an inequality greater than 0: {cond}."
                    "Check thanwerdas2022 theorem 8.7"
                )
            return

        cond_1 = 2 * alpha + (n - 2) * beta
        cond_2 = alpha + (n - 1) * (beta + n * gamma)
        if cond_1 < gs.atol or cond_2 < gs.atol:
            raise ValueError(
                f"Inequalities should be greater than 0, but: {cond_1} and {cond_2}."
                "Check thanwerdas2022 theorem 8.7"
            )

    def _quadratic_form(self, tangent_vec):
        """Quadratic form associated to inner product.

        Parameters
        ----------
        tangent_vec: array-like, shape=[..., n, n]
            Tangent vector at base point.
        """
        comp = gs.matmul(tangent_vec, tangent_vec)
        out_alpha = self.alpha * gs.trace(comp) if self.alpha > gs.atol else 0.0
        out_beta = (
            self.beta * gs.sum(comp, axis=(-2, -1)) if self.beta > gs.atol else 0.0
        )
        out_gamma = self.gamma * gs.sum(tangent_vec, axis=(-2, -1)) ** 2

        return out_alpha + out_beta + out_gamma

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point: array-like, shape=[..., n, n]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        inner_prod = (1 / 2) * (
            self._quadratic_form(tangent_vec_a + tangent_vec_b)
            - self._quadratic_form(tangent_vec_a)
            - self._quadratic_form(tangent_vec_b)
        )
        return repeat_out(
            self._space.point_ndim, inner_prod, tangent_vec_a, tangent_vec_b, base_point
        )

    def squared_norm(self, vector, base_point=None):
        """Compute the square of the norm of a vector.

        Squared norm of a vector associated to the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Vector.
        base_point : array-like, shape=[..., n, n]
            Base point.
            Optional, default: None.

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm.
        """
        return self._quadratic_form(vector)


class ConstantValueRowSumsDiffeo(Diffeo):
    r"""A diffeomorphism from the constant-value-row-sum matrices to symmetric matrices.

    A particular case is the diffeomorphism between the space of null-row-sum symmetric
    n-matrices and the space of symmetric (n-1)-matrices.

    Let :math:`f` be the diffeomorphism
    :math:`f: M \rightarrow N` of the manifold
    :math:`M` into the manifold :math:`N`.
    """

    def __init__(self, value=0.0):
        self.value = value
        self._space_ndim = 2
        self._image_space_ndim = 2

    def diffeomorphism(self, base_point):
        """Diffeomorphism at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., n-1, n-1]
            Image point.
        """
        return base_point[..., :-1, :-1]

    def _concatenate_row_sums(self, image_point, value):
        """Concatenate missing row and column."""
        row_sums = value - gs.sum(image_point, axis=-1)
        last_row = gs.concatenate(
            [row_sums, gs.expand_dims(value - gs.sum(row_sums, axis=-1), axis=-1)],
            axis=-1,
        )
        return gs.concatenate(
            [
                gs.concatenate(
                    [image_point, gs.expand_dims(row_sums, axis=-1)], axis=-1
                ),
                gs.expand_dims(last_row, axis=-2),
            ],
            axis=-2,
        )

    def inverse_diffeomorphism(self, image_point):
        r"""Inverse diffeomorphism at image point.

        :math:`f^{-1}: N \rightarrow M`

        Parameters
        ----------
        image_point : array-like, shape=[..., n-1, n-1]
            Image point.

        Returns
        -------
        base_point : array-like, shape=[..., n, n]
            Base point.
        """
        return self._concatenate_row_sums(image_point, self.value)

    def tangent_diffeomorphism(self, tangent_vec, base_point=None, image_point=None):
        r"""Tangent diffeomorphism at base point.

        df_p is a linear map from T_pM to T_f(p)N.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image of the base point.
        """
        out = self.diffeomorphism(tangent_vec)
        return repeat_out_multiple_ndim(
            out,
            self._space_ndim,
            (tangent_vec, base_point),
            self._image_space_ndim,
            (image_point,),
            out_ndim=self._image_space_ndim,
        )

    def inverse_tangent_diffeomorphism(
        self, image_tangent_vec, image_point=None, base_point=None
    ):
        r"""Inverse tangent diffeomorphism at image point.

        df^-1_p is a linear map from T_f(p)N to T_pM

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        """
        out = self._concatenate_row_sums(image_tangent_vec, 0.0)
        return repeat_out_multiple_ndim(
            out,
            self._image_space_ndim,
            (image_tangent_vec, image_point),
            self._space_ndim,
            (base_point,),
            out_ndim=self._space_ndim,
        )


class NullRowSumsSymmetricMatrices(LevelSet, DiffeomorphicMatrixVectorSpace):
    r"""Space of null-row-sums symmetric matrices.

    Set of symmetric matrices with null row sums:

    .. math::

        \operatorname{Row_0}(n) = \{S \in \operatorname{Sym}(n)
        \mid S \mathbb{1}=0 \}

    Check out chapter 8 of [T2022]_ and [T2023]_ for more details.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.

    References
    ----------
    .. [T2022] Yann Thanwerdas. Riemannian and stratified
        geometries on covariance and correlation matrices. Differential
        Geometry [math.DG]. Université Côte d'Azur, 2022.
    .. [T2023] Thanwerdas, Yann. “Permutation-Invariant Log-Euclidean Geometries
        on Full-Rank Correlation Matrices,”
        November 2023. https://hal.science/hal-03878729.
    """

    def __init__(self, n, equip=True):
        self.n = n
        image_space = SymmetricMatrices(n - 1, equip=False)
        super().__init__(
            dim=image_space.dim,
            diffeo=ConstantValueRowSumsDiffeo(value=0.0),
            image_space=image_space,
            shape=(n, n),
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MatricesMetric

    def _define_embedding_space(self):
        """Define embedding space of the manifold.

        Returns
        -------
        embedding_space : Manifold
            Instance of Manifold.
        """
        return SymmetricMatrices(n=self.n)

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]

        Returns
        -------
        submersed_point : array-like, shape=[..., n]
        """
        return gs.sum(point, axis=-1)

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
        point : Ignored.

        Returns
        -------
        submersed_vector : array-like, shape=[..., n]
        """
        out = self.submersion(vector)
        return repeat_out(self.point_ndim, out, vector, point, out_shape=(self.n,))

    def _create_basis(self):
        """Compute the basis of the vector space."""
        indices, values = [], []
        k = -1
        for row in range(self.n - 1):
            for col in range(row, self.n - 1):
                k += 1
                if row == col:
                    indices.append((k, row, row))
                    values.append(1.0)
                else:
                    indices.extend([(k, row, col), (k, col, row)])
                    values.extend([1.0, 1.0])

        pre_basis = gs.array_from_sparse(indices, values, (k + 1, self.n, self.n))
        return self.matrix_representation(
            self.basis_representation(
                pre_basis,
            )
        )


class NullRowSumsPermutationInvariantMetric(EuclideanMetric):
    r"""A permutation-invariant metric on the space of null-row-sums symmetric matrices.

    It is flat Riemannian metric invariant by the congruence action
    of permutation matrices defined over a matrix vector space.

    Its associated quadratic form is:

    .. math::

        q(Y)=\alpha \operatorname{tr}\left(Y^2\right)
        +\delta \operatorname{tr}\left(\operatorname{Diag}(Y)^2\right)
        +\zeta \operatorname{tr}(Y)^2

    Check out chapter 8 of [T2022]_ and [T2023]_ for more details.

    Parameters
    ----------
    space : Manifold
    alpha : float
        Scalar multiplying first term of quadratic form.
    delta : float
        Scalar multiplying second term of quadratic form.
    zeta : float
        Scalar multiplying third term of quadratic form.

    References
    ----------
    .. [T2022] Yann Thanwerdas. Riemannian and stratified
        geometries on covariance and correlation matrices. Differential
        Geometry [math.DG]. Université Côte d'Azur, 2022.
    .. [T2023] Thanwerdas, Yann. “Permutation-Invariant Log-Euclidean Geometries
        on Full-Rank Correlation Matrices,”
        November 2023. https://hal.science/hal-03878729.
    """

    def __init__(self, space, alpha=1.0, delta=1.0, zeta=1.0):
        self._check_params(space, alpha, delta, zeta)
        super().__init__(space=space)
        self.alpha = alpha
        self.delta = delta
        self.zeta = zeta

    @staticmethod
    def _check_params(space, alpha, delta, zeta):
        r"""Check parameters of quadratic form.

        The following conditions must verify:
        - n > 3: :math:`\alpha>0, n\alpha+(n-2)\delta>0, n\alpha+(n-1)(\delta+n\zeta)>0`
        - n = 3: :math:`\alpha=0, \delta > 0, \delta + 3 \zeta > 0`
        - n = 2: :math:`\alpha=\delta=0, \zeta > 0`
        """
        n = space.n
        if n == 2:
            if alpha > gs.atol or delta > gs.atol or zeta < gs.atol:
                raise ValueError(
                    f"When n==2: alpha ({alpha}) and delta({delta}) must be 0,"
                    f"and zeta ({zeta}) > 0. "
                )
            return

        elif n == 3:
            cond = delta + 3 * zeta
            if alpha > gs.atol or delta < gs.atol or cond < gs.atol:
                raise ValueError(
                    f"When n==3: alpha ({alpha}) must be 0, delta ({delta}) > 0"
                    f"and an inequality greater than 0: {cond}."
                    "Check thanwerdas2022 theorem 8.7"
                )
            return

        cond_1 = n * alpha + (n - 2) * delta
        cond_2 = n * alpha + (n - 1) * (delta + n * zeta)
        if cond_1 < gs.atol or cond_2 < gs.atol:
            raise ValueError(
                f"Inequalities should be greater than 0, but: {cond_1} and {cond_2}."
                "Check thanwerdas2022 theorem 8.7"
            )

    def _quadratic_form(self, tangent_vec):
        """Quadratic form associated to inner product.

        Parameters
        ----------
        tangent_vec: array-like, shape=[..., n, n]
            Tangent vector at base point.
        """
        if self.alpha > gs.atol:
            comp = gs.matmul(tangent_vec, tangent_vec)
            out_alpha = self.alpha * gs.trace(comp)
        else:
            out_alpha = 0.0
        out_delta = (
            self.delta * gs.sum(Matrices.diagonal(tangent_vec) ** 2, axis=-1)
            if self.delta > gs.atol
            else 0.0
        )
        out_zeta = self.zeta * gs.trace(tangent_vec) ** 2

        return out_alpha + out_delta + out_zeta

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point: array-like, shape=[..., n, n]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        inner_prod = (1 / 2) * (
            self._quadratic_form(tangent_vec_a + tangent_vec_b)
            - self._quadratic_form(tangent_vec_a)
            - self._quadratic_form(tangent_vec_b)
        )
        return repeat_out(
            self._space.point_ndim, inner_prod, tangent_vec_a, tangent_vec_b, base_point
        )

    def squared_norm(self, vector, base_point=None):
        """Compute the square of the norm of a vector.

        Squared norm of a vector associated to the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Vector.
        base_point : array-like, shape=[..., n, n]
            Base point.
            Optional, default: None.

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm.
        """
        return self._quadratic_form(vector)
