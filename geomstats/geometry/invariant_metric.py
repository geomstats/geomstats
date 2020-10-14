"""Left- and right- invariant metrics that exist on Lie groups."""

import logging

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric


class InvariantMetric(RiemannianMetric):
    """Class for invariant metrics which exist on Lie groups.

    This class supports both left and right invariant metrics
    which exist on Lie groups.

    Points are parameterized by the Riemannian logarithm
    for the canonical left-invariant metric.

    Parameters
    ----------
    group : LieGroup
        Group to equip with the invariant metric
    metric_mat_at_identity : array-like, shape=[dim, dim]
        Matrix that defines the metric at identity.
        Optional, defaults to identity matrix if None.
    left_or_right : str, {'left', 'right'}
        Wether to use a left or right invariant metric.
        Optional, default: 'left'.
    """

    def __init__(self, group, algebra=None,
                 metric_mat_at_identity=None,
                 left_or_right='left', **kwargs):
        super(InvariantMetric, self).__init__(dim=group.dim, **kwargs)

        self.group = group
        self.lie_algebra = algebra
        if metric_mat_at_identity is None:
            metric_mat_at_identity = gs.eye(self.group.dim)

        geomstats.errors.check_parameter_accepted_values(
            left_or_right, 'left_or_right', ['left', 'right'])

        eigenvalues = gs.linalg.eigvalsh(metric_mat_at_identity)
        mask_pos_eigval = gs.greater(eigenvalues, 0.)
        n_pos_eigval = gs.sum(gs.cast(mask_pos_eigval, gs.int32))
        mask_neg_eigval = gs.less(eigenvalues, 0.)
        n_neg_eigval = gs.sum(gs.cast(mask_neg_eigval, gs.int32))
        mask_null_eigval = gs.isclose(eigenvalues, 0.)
        n_null_eigval = gs.sum(gs.cast(mask_null_eigval, gs.int32))

        self.metric_mat_at_identity = metric_mat_at_identity
        self.left_or_right = left_or_right
        self.signature = (n_pos_eigval, n_null_eigval, n_neg_eigval)

    def inner_product_at_identity(self, tangent_vec_a, tangent_vec_b):
        """Compute inner product at tangent space at identity.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            First tangent vector at identity.
        tangent_vec_b : array-like, shape=[..., dim]
            Second tangent vector at identity.

        Returns
        -------
        inner_prod : array-like, shape=[..., dim]
            Inner-product of the two tangent vectors.
        """
        geomstats.errors.check_parameter_accepted_values(
            self.group.default_point_type,
            'default_point_type',
            ['vector', 'matrix'])

        if self.group.default_point_type == 'vector':
            inner_product_mat_at_identity = self.metric_mat_at_identity
            inner_prod = gs.einsum(
                '...i,...ij->...j',
                tangent_vec_a,
                inner_product_mat_at_identity)
            inner_prod = gs.einsum(
                '...j,...j->...', inner_prod, tangent_vec_b)

        else:
            logging.warning(
                'The inner product'
                ' is only implemented for diagonal inner-product matrices, '
                'and the Lie Algebra needs to be specified.')
            is_vectorized = \
                (gs.ndim(tangent_vec_a) == 3) or (gs.ndim(tangent_vec_b) == 3)
            axes = (2, 1) if is_vectorized else (0, 1)

            aux_prod = tangent_vec_a * tangent_vec_b
            metric_mat = self.metric_mat_at_identity
            if (Matrices.is_diagonal(metric_mat)
                    and self.lie_algebra is not None):
                aux_prod *= self.lie_algebra.reshape_metric_matrix(
                    metric_mat)
            inner_prod = gs.sum(aux_prod, axis=axes)

        return inner_prod

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute inner product of two vectors in tangent space at base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            First tangent vector at base_point.
        tangent_vec_b : array-like, shape=[..., dim]
            Second tangent vector at base_point.
        base_point : array-like, shape=[..., dim]
            Point in the group.
            Optional, defaults to identity if None.

        Returns
        -------
        inner_prod : array-like, shape=[..., dim]
            Inner-product of the two tangent vectors.
        """
        if base_point is None:
            return self.inner_product_at_identity(
                tangent_vec_a, tangent_vec_b)
        if self.group.default_point_type == 'vector':
            return super(InvariantMetric, self).inner_product(
                tangent_vec_a,
                tangent_vec_b,
                base_point)

        tangent_translation = self.group.tangent_translation_map(
            base_point, left_or_right=self.left_or_right, inverse=True)
        tangent_vec_a_at_id = tangent_translation(tangent_vec_a)
        tangent_vec_b_at_id = tangent_translation(tangent_vec_b)
        inner_prod = self.inner_product_at_identity(
            tangent_vec_a_at_id, tangent_vec_b_at_id)
        return inner_prod

    def metric_matrix(self, base_point=None):
        """Compute inner product matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim], optional
            Point in the group (the default is identity).

        Returns
        -------
        metric_mat : array-like, shape=[..., dim, dim]
            Metric matrix at base_point.
        """
        if self.group.default_point_type == 'matrix':
            raise NotImplementedError(
                'inner_product_matrix not implemented for Lie groups'
                ' whose elements are represented as matrices.')

        if base_point is None:
            base_point = self.group.identity
        else:
            base_point = self.group.regularize(base_point)

        jacobian = self.group.jacobian_translation(
            point=base_point, left_or_right=self.left_or_right)

        inv_jacobian = GeneralLinear.inverse(jacobian)
        inv_jacobian_transposed = Matrices.transpose(inv_jacobian)

        metric_mat = gs.einsum(
            '...ij,...jk->...ik',
            inv_jacobian_transposed, self.metric_mat_at_identity)
        metric_mat = gs.einsum(
            '...ij,...jk->...ik', metric_mat, inv_jacobian)
        return metric_mat

    def structure_constant(self, tan_a, tan_b, tan_c):
        """
        <[x,y],z>
        """
        return self.inner_product_at_identity(
            GeneralLinear.bracket(tan_a, tan_b), tan_c)

    def adjoint_star(self, tan_a, tan_b):
        """
        <ad(x)*y, z> = <[x,z], y > pour tout z
        """
        basis = self.lie_algebra.orthonormal_basis(
            self.metric_mat_at_identity)
        return - gs.einsum(
            'i...,ijk->...jk',
            gs.array([
                self.structure_constant(tan, tan_a, tan_b) for tan in basis]),
            gs.array(basis))

    def connection_at_identity(self, tan_a_at_id, tan_b_at_id):
        return 1. / 2 * (GeneralLinear.bracket(tan_a_at_id, tan_b_at_id)
                         - self.adjoint_star(tan_a_at_id, tan_b_at_id)
                         - self.adjoint_star(tan_b_at_id, tan_a_at_id))

    def connection(self, tangent_vec_a, tangent_vec_b, base_point=None):
        if base_point is None:
            return self.connection_at_identity(tangent_vec_a, tangent_vec_b)
        translation_map = self.group.tangent_translation_map(
            base_point, left_or_right=self.left_or_right, inverse=True)
        tan_a_at_id = translation_map(tangent_vec_a)
        tan_b_at_id = translation_map(tangent_vec_b)
        return self.connection_at_identity(tan_a_at_id, tan_b_at_id)

    def curvature(self, x, y, z):
        """
        R(x, y)z
        """
        return (
                self.connection_at_identity(GeneralLinear.bracket(x, y), z)
                - self.connection_at_identity(
            x, self.connection_at_identity(y, z))
                + self.connection_at_identity(
            y, self.connection_at_identity(x, z)))

    def sectional_curvature(self, x, y):
        """
        < R(x, y)x, y> for x, y orthogonal. This is compensated if not
        """
        num = self.inner_product(y, self.curvature(x, y, x))
        denom = (
                self.inner_product(x, x)
                * self.inner_product(y, y)
                - self.inner_product(x, y) ** 2)
        condition = gs.isclose(denom, 0.)
        return gs.divide(num, denom, where=~condition)

    def sectional_curvature_at_point(
            self, tangent_vec_a, tangent_vec_b, base_point=None):
        if base_point is None:
            return self.sectional_curvature(tangent_vec_a, tangent_vec_b)
        translation_map = self.group.tangent_translation_map(
            base_point, inverse=True, left_or_right=self.left_or_right)
        tan_a_at_id = translation_map(tangent_vec_a)
        tan_b_at_id = translation_map(tangent_vec_b)
        return self.sectional_curvature(tan_a_at_id, tan_b_at_id)

    def left_exp_from_identity(self, tangent_vec):
        """Compute the exponential from identity with the left-invariant metric.

        Compute Riemannian exponential of a tangent vector at the identity
        associated to the left-invariant metric.

        If the method is called by a right-invariant metric, it uses the
        left-invariant metric associated to the same inner-product matrix
        at the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at identity.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point in the group.
        """
        tangent_vec = self.group.regularize_tangent_vec_at_identity(
            tangent_vec=tangent_vec,
            metric=self)
        sqrt_inner_product_mat = gs.linalg.sqrtm(
            self.metric_mat_at_identity)
        mat = Matrices.transpose(sqrt_inner_product_mat)

        exp = gs.einsum('...i,...ij->...j', tangent_vec, mat)

        exp = self.group.regularize(exp)
        return exp

    def exp_from_identity(self, tangent_vec):
        """Compute Riemannian exponential of tangent vector from the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at identity.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point in the group.
        """
        if self.left_or_right == 'left':
            exp = self.left_exp_from_identity(tangent_vec)

        else:
            opp_left_exp = self.left_exp_from_identity(-tangent_vec)
            exp = self.group.inverse(opp_left_exp)

        exp = self.group.regularize(exp)
        return exp

    def exp(self, tangent_vec, base_point=None):
        """Compute Riemannian exponential of tan. vector wrt to base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim]
            Point in the group.
            Optional, defaults to identity if None.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point in the group equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        identity = self.group.identity

        if base_point is None:
            base_point = identity
        base_point = self.group.regularize(base_point)

        if gs.allclose(base_point, identity):
            return self.exp_from_identity(tangent_vec)

        tangent_vec_at_id = self.group.tangent_translation_map(
            point=base_point,
            left_or_right=self.left_or_right,
            inverse=True)(tangent_vec)
        exp_from_id = self.exp_from_identity(tangent_vec_at_id)

        if self.left_or_right == 'left':
            exp = self.group.compose(base_point, exp_from_id)

        else:
            exp = self.group.compose(exp_from_id, base_point)

        exp = self.group.regularize(exp)

        return exp

    def left_log_from_identity(self, point):
        """Compute Riemannian log of a point wrt. id of left-invar. metric.

        Compute Riemannian logarithm of a point wrt the identity associated
        to the left-invariant metric.

        If the method is called by a right-invariant metric, it uses the
        left-invariant metric associated to the same inner-product matrix
        at the identity.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in the group.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the identity equal to the Riemannian logarithm
            of point at the identity.
        """
        point = self.group.regularize(point)
        inner_prod_mat = self.metric_mat_at_identity
        inv_inner_prod_mat = GeneralLinear.inverse(inner_prod_mat)
        sqrt_inv_inner_prod_mat = gs.linalg.sqrtm(inv_inner_prod_mat)
        log = gs.einsum('...i,...ij->...j', point, sqrt_inv_inner_prod_mat)
        log = self.group.regularize_tangent_vec_at_identity(
            tangent_vec=log, metric=self)
        return log

    def log_from_identity(self, point):
        """Compute Riemannian logarithm of a point wrt the identity.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in the group.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the identity equal to the Riemannian logarithm
            of point at the identity.
        """
        point = self.group.regularize(point)
        if self.left_or_right == 'left':
            log = self.left_log_from_identity(point)

        else:
            inv_point = self.group.inverse(point)
            left_log = self.left_log_from_identity(inv_point)
            log = - left_log

        return log

    def log(self, point, base_point=None):
        """Compute Riemannian logarithm of a point from a base point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in the group.
        base_point : array-like, shape=[..., dim], optional
            Point in the group, from which to compute the log,
            (the default is identity).

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        identity = self.group.identity

        if base_point is None:
            base_point = identity
        base_point = self.group.regularize(base_point)

        if gs.allclose(base_point, identity):
            return self.log_from_identity(point)

        point = self.group.regularize(point)

        if self.left_or_right == 'left':
            point_near_id = self.group.compose(
                self.group.inverse(base_point), point)

        else:
            point_near_id = self.group.compose(
                point, self.group.inverse(base_point))

        log_from_id = self.log_from_identity(point_near_id)
        log = self.group.tangent_translation_map(
            base_point, left_or_right=self.left_or_right)(log_from_id)
        return log


class BiInvariantMetric(InvariantMetric):
    """Class for bi-invariant metrics which exist on Lie groups.

    Compact Lie groups and direct products of compact Lie groups with vector
    spaces admit bi-invariant metrics. Products Lie groups are not
    implemented. Other groups such as SE(3) admit bi-invariant pseudo-metrics.

    Parameters
    ----------
    group : LieGroup
        The group to equip with the bi-invariant metric
    """

    def __init__(self, group):
        super(BiInvariantMetric, self).__init__(group=group,
                                                metric_mat_at_identity=gs.eye(
                                                    group.dim),
                                                default_point_type=group.default_point_type)
        cond = (
            'SpecialOrthogonal' not in group.__str__()
            and 'SO' not in group.__str__()
            and 'SpecialOrthogonal3' not in group.__str__())
        # TODO (nguigs): implement it for SE(3)
        if cond:
            raise ValueError(
                'The bi-invariant metric is only implemented for SO(n)')

    def exp_from_identity(self, tangent_vec):
        """Compute Riemannian exponential of tangent vector from the identity.

        For a bi-invariant metric, this corresponds to the group exponential.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at identity.

        Returns
        -------
        exp : array-like, shape=[..., {dim, [n, n]}]
            Point in the group.
        """
        return self.group.exp(tangent_vec)

    def log_from_identity(self, point):
        """Compute Riemannian logarithm of a point wrt the identity.

        For a bi-invariant metric this corresponds to the group logarithm.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
            Point in the group.

        Returns
        -------
        log : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at the identity equal to the Riemannian logarithm
            of point at the identity.
        """
        return self.group.log(point)
