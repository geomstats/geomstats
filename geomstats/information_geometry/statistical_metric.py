"""Statistical metric and conjugate connections to equip manifold."""

from typing import Any
import geomstats.backend as gs
from geomstats.geometry.connection import Connection
from geomstats.geometry.riemannian_metric import RiemannianMetric


def unpack_inputs(func, dim):
    r"""Wrap function to unpack inputs to divergence function.

    Compose a function :math:`g:` with a wrapper :math:`f`
    to identify
    .. math::
        g: \mathbb{R}^{n} \times \mathbb{R}^{n} \to \mathbb{R}

    with the function
    .. math::
        \tidle{g} = f\circ g: \mathbb{R}^{2n} \to \mathbb{R}

    Autodifferentiation is then computed on :math:`\tidle{g}`.

    Parameters
    ----------
    func : callable
        Function to unpack inputs.
    dim : int
        Dimension of the manifold.

    Returns
    -------
    callable
        Composition of wrapper function with original function.
    """
    def wrapper(tensor):
        return func(tensor[..., : dim], tensor[..., dim :])
    return wrapper


class DivergenceConnection(Connection):
    r"""Class to derive a connection from a divergence.

    When implemented in conjuction with the DualDivergenceConnection object,
    a pair of conjugate connections can be defined on a manifold.

    Given an :math:`n` manifold :math:`M` with coordinates :math:`x`
    and a divergence :math:`D: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}`,
    the conjugate connections :math:`\nabla^D` and :math:`\nabla^{D^*}` can be defined
    in terms of their corresponding Christoffel symbols :math:`\Gamma^D`
    and :math:`\Gamma^{D^*}`.

    These connections are said to be conjugate with respect to the divergence induced
    metric :math:`g^D` because for any smooth vector fields :math:`X, Y, Z`
    on :math:`M` we have:

    .. math::
        X(g^D(Y, Z)) = g^D(\nabla^D_X Y, Z) + g^D(Y, \nabla^{D^*}_X Z)

    The Levi-Civita connection can be recovered from these connections:

    .. math::
        \nabla^{LC} = \frac{\nabla^D + \nabla^{D^*}}{2}

    Attributes
    ----------
    space : Manifold
        Manifold to equip with the divergence and its conjugate connections.
    divergence : callable
        Divergence function, takes two inputs of shape=[..., dim] and returns a scalar.

    References
    ----------
    .. [N2020] F. Nielsen,
        "An Elementary Introduction to Information Geometry",
        arXiv:808.08271v2 (2020): 10-16
    """

    def __init__(self, space, divergence):
        super().__init__(space=space)
        assert (
            space.intrinsic
        ), "The manifold must be parametrized by an intrinsic coordinate system."
        self.dim = space.dim
        self.divergence = unpack_inputs(divergence, self.dim)

    def first_kind_christoffels(self, base_point):
        r"""Compute the first kind Christoffel symbols of the divergence connection.

        Compute the first kind Christoffel symbols of the divergence connection
        :math:`\nabla^D` at the tangent space of the base point.

        .. math::
            \Gamma^D_{i j k} =
            -1 \cdot \frac{\partial^2}{\partial x^i \partial x^j}
                \frac{\partial}{\partial y^k} D(x, y) \bigg|_{x=y}

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim, dim]
            First kind Christoffel symbols of the divergence connection.
        """
        base_point_pair = gs.concatenate([base_point, base_point])
        hess = gs.autodiff.hessian(self.divergence)
        jac_hess = gs.autodiff.jacobian(hess)
        first_kind = jac_hess(base_point_pair)[: self.dim, : self.dim, self.dim :]
        first_kind = -1 * first_kind
        return first_kind

    def christoffels(self, base_point):
        r"""Compute the (second kind) Christoffel symbols of the divergence connection.

        Compute the (second kind) Christoffel symbols of the divergence connection
        :math:`\nabla^D` at the tangent space of the base point.

        .. math::
            \Gamma^D_{ij}^k = g^{kl} \Gamma^D_{ijl}

        where :math:`g^{kl}` is the cometric matrix of the divergence induced metric.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim, dim]
            Second kind Christoffel symbols of the divergence connection.
        """
        base_point_pair = gs.concatenate([base_point, base_point])
        metric_hess = gs.autodiff.hessian(self.divergence)
        metric_matrix = -1 * metric_hess(base_point_pair)[: self.dim, self.dim :]
        cometric_matrix = gs.linalg.inv(metric_matrix)
        first_kind_christoffels = self.first_kind_christoffels(base_point)
        second_kind_christoffels = gs.einsum(
            '...lk, ...ijl -> ...kij', cometric_matrix, first_kind_christoffels
        )
        return second_kind_christoffels


class DualDivergenceConnection(Connection):
    r"""Class to derive a dual connection from a divergence.

    When implemented in conjuction with the DivergenceConnection object,
    a pair of conjugate connections can be defined on a manifold.

    Given an :math:`n` manifold :math:`M` with coordinates :math:`x`
    and a divergence :math:`D: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}`,
    the conjugate connections :math:`\nabla^D` and :math:`\nabla^{D^*}` can be defined
    in terms of their corresponding Christoffel symbols :math:`\Gamma^D`
    and :math:`\Gamma^{D^*}`.

    These connections are said to be conjugate with respect to the divergence induced
    metric :math:`g^D` because for any smooth vector fields :math:`X, Y, Z`
    on :math:`M` we have:

    .. math::
        X(g^D(Y, Z)) = g^D(\nabla^D_X Y, Z) + g^D(Y, \nabla^{D^*}_X Z)

    The Levi-Civita connection can be recovered from these connections:

    .. math::
        \nabla^{LC} = \frac{\nabla^D + \nabla^{D^*}}{2}

    Attributes
    ----------
    space : Manifold
        Manifold to equip with the divergence and its conjugate connections.
    divergence : callable
        Divergence function, takes two inputs of shape=[..., dim] and returns a scalar.

    References
    ----------
    .. [N2020] F. Nielsen,
        "An Elementary Introduction to Information Geometry",
        arXiv:808.08271v2 (2020): 10-16
    """

    def __init__(self, space, divergence):
        super().__init__(space=space)
        assert (
            space.intrinsic
        ), "The manifold must be parametrized by an intrinsic coordinate system."
        self.dim = space.dim
        self.divergence = unpack_inputs(divergence, self.dim)

    def first_kind_christoffels(self, base_point):
        r"""Compute the first kind Christoffel symbols of dual divergence connection.

        Compute the first kind Christoffel symbols of the dual divergence connection
        :math:`\nabla^{D^*}` at the tangent space of the base point.

        .. math::
            \Gamma^{D^*}_{i j k} =
            -1 \cdot \frac{\partial^2}{\partial y^i \partial y^j}
                \frac{\partial}{\partial x^k} D(x, y) \bigg|_{x=y}

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim, dim]
            First kind Christoffel symbols of the dual divergence connection.
        """
        base_point_pair = gs.concatenate([base_point, base_point])
        hess = gs.autodiff.hessian(self.divergence)
        jac_hess = gs.autodiff.jacobian(hess)
        first_kind = jac_hess(base_point_pair)[self.dim :, self.dim :, : self.dim]
        first_kind = -1 * first_kind
        return first_kind

    def christoffels(self, base_point):
        r"""Compute the (second kind) Christoffel symbols of dual divergence connection.

        Compute the (second kind) Christoffel symbols of the dual divergence connection
        :math:`\nabla^{D^*}` at the tangent space of the base point.

        .. math::
            \Gamma^{D^*}_{ij}^k = g^{kl} \Gamma^{D^*}_{ijl}

        where :math:`g^{kl}` is the cometric matrix of the divergence induced metric.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim, dim]
            Second kind Christoffel symbols of the dual divergence connection.
        """
        base_point_pair = gs.concatenate([base_point, base_point])
        metric_hess = gs.autodiff.hessian(self.divergence)
        metric_matrix = -1 * metric_hess(base_point_pair)[: self.dim, self.dim :]
        cometric_matrix = gs.linalg.inv(metric_matrix)
        first_kind_christoffels = self.first_kind_christoffels(base_point)
        second_kind_christoffels = gs.einsum(
            '...lk, ...ijl -> ...kij',
            cometric_matrix,
            first_kind_christoffels
        )
        return second_kind_christoffels


class AlphaConnection(Connection):
    r"""Class to define a connection that interpolates between conjugate connections.

    Given a pair of conjugate connections :math:`\nabla^D` and :math:`\nabla^{D^*}`
    on a manifold :math:`M`, a connection :math:`\nabla^{\alpha}` can be defined
    that interpolates between the two connections.

    The connection :math:`\nabla^{\alpha}` is defined in terms of the Christoffel
    symbols :math:`\Gamma^D` and :math:`\Gamma^{D^*}` of the conjugate connections.

    .. math::
        \Gamma^{\alpha}_{i j k} = (1 + \alpha)/2 \Gamma^D_{i j k}
            + (1 - \alpha)/2 \Gamma^{D^*}_{i j k}

    When :math:`\alpha = 0`, the connection :math:`\nabla^{\alpha}` reduces to the
    Levi-Civita connection :math:`\nabla^{LC}`.

    A family of :math:`\alpha` conjugate connections can be defined on a manifold by the
    relation :math:`\nabla^{\alpha} = (\nabla^{-\alpha})^*`.

    Attributes
    ----------
    space : Manifold
        Manifold to equip with the divergence and its conjugate connections.
    alpha : float
        Interpolation parameter.
    primal_connection : Connection
        Primal connection to interpolate between.
    dual_connection : Connection
        Dual connection to interpolate between.

    References
    ----------
    .. [N2020] F. Nielsen,
        "An Elementary Introduction to Information Geometry",
        arXiv:808.08271v2 (2020): 10-16
    """

    def __init__(self, space, alpha, primal_connection, dual_connection):
        super().__init__(space=space)
        self.alpha = alpha
        self.primal_connection = primal_connection
        self.dual_connection = dual_connection

    def christoffels(self, base_point):
        r"""Compute the Christoffel symbols of the alpha connection.

        Compute the Christoffel symbols of the alpha connection :math:`\nabla^{\alpha}`
        at the tangent space of the base point.

        .. math::
            \Gamma^{\alpha}_{i j k} = (1 + \alpha)/2 \Gamma^D_{i j k}
                + (1 - \alpha)/2 \Gamma^{D^*}_{i j k}

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim, dim]
            Christoffel symbols of the alpha connection.
        """
        primal_christoffels = self.primal_connection.christoffels(base_point)
        dual_christoffels = self.dual_connection.christoffels(base_point)
        primal_part = primal_christoffels * (1 + self.alpha) / 2
        dual_part = dual_christoffels * (1 - self.alpha) / 2
        return primal_part + dual_part


class StatisticalMetric(RiemannianMetric):
    r"""Class to define a statistical metric on a manifold.

    Given an :math:`n` manifold :math:`M` with coordinates :math:`x` and
    a divergence :math:`D: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}`,
    :math:`M` can be equipped with a divergence induced  metric :math:`g^D`
    and  a (0, 3) tensor :math:`C^D` called the Amari divergence tensor by
    way of a pair of conjugate connections :math:`\nabla^D` and
    :math:`\nabla^{D^*}`.

    The corresponding statistical manifold is therefore
    .. math::
        (M, g^D, C^D)

    Attributes
    ----------
    space : Manifold
        Manifold to equip with the divergence induced metric.
    divergence : callable
        Divergence function, takes two inputs of shape=[..., dim] and returns a scalar.
    primal_connection : Connection
        Primal connection to define the divergence induced metric.
    dual_connection : Connection
        Dual connection to define the divergence induced metric.

    References
    ----------
    .. [N2020] F. Nielsen,
        "An Elementary Introduction to Information Geometry",
        arXiv:808.08271v2 (2020): 10-16
    .. [A1985] S. Amari, (1985)
            Differential Geometric Methods in Statistics, Springer.
    """

    def __init__(self, space, divergence, primal_connection, dual_connection):
        super().__init__(space=space, signature=(space.dim, 0))
        assert (
            space.intrinsic
        ), "The manifold must be parametrized by an intrinsic coordinate system."

        self.dim = space.dim
        self.divergence = unpack_inputs(divergence, self.dim)
        self.primal_connection = primal_connection
        self.dual_connection = dual_connection

    def metric_matrix(self, base_point):
        r"""Compute the divergence induced metric matrix.

        Compute the inner-product matrix of the divergence induced metric
        at the tangent space at the base point.

        .. math::
            g^D_{i,j}(x) =
            -1 \cdot \frac{\partial}{\parital x^i}
                \frac{\partial}{\partial y^j} D(x, y) \bigg|_{x=y}

        In the case of Kullback-Leibler divergence, the Fisher-Rao metric is recovered.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim]
            Inner-product matrix of the divergence induced metric.

        See Also
        --------
        geomstats.geometry.fisher_rao_metric.FisherRaoMetric
        """
        hess = gs.autodiff.hessian(self.divergence)
        base_point_pair = gs.concatenate([base_point, base_point])
        return -1 * hess(base_point_pair)[: self.dim, self.dim :]

    def amari_divergence_tensor(self, base_point):
        r"""Compute the Amari divergence tensor.

        Compute the Amari divergence tensor :math:`C^D` of the divergence induced metric
        at the tangent space at the base point.

        .. math::
            C^D_{i j k} = \Gamma^{D^*}_{i j k} - \Gamma^D_{i j k}

        where :math:`\Gamma^D` and :math:`\Gamma^{D^*}` are the Christoffel symbols
        corresponding to the divergence induced conjugate connections.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        tensor : array-like, shape=[..., dim, dim, dim]
            Amari divergence tensor.
        """
        first_primal = self.primal_connection.first_kind_christoffels(base_point)
        first_dual = self.dual_connection.first_kind_christoffels(base_point)
        return first_dual - first_primal

class PotentialFunction:
    def __call__(self, x):
        raise NotImplementedError

    def hessian(self, point):
        return gs.autodiff.hessian(self.__call__)(point)

    def third_derivative(self, point):
        return gs.autodiff.jacobian(self.hessian)(point)

class StatisticalMetricFromPotentialFunction(RiemannianMetric):
    def __init__(self, space, potential_function):
        super().__init__(space=space, signature=(space.dim, 0))
        self.dim = space.dim
        self.potential_function = potential_function

    def metric_matrix(self, base_point):
        return self.potential_function.hessian(base_point)
    
    def first_kind_christoffels(self, base_point):
        christoffels = self.christoffels(base_point)
        cometric = self.cometric_matrix(base_point)
        return gs.einsum('...lk, ...lij -> ...kij', cometric, christoffels)
    
    def amari_chentsov_tensor(self, base_point):
        return self.potential_function.third_derivative(base_point)
    
class AlphaConnectionFromPotentialFunction(Connection):
    def __init__(self, statstical_metric_from_func, alpha=1):
        self.stat_metric = statstical_metric_from_func
        self.alpha = alpha

    def first_kind_christoffels(self, base_point):
        LC_first_kind = self.stat_metric.first_kind_christoffels(base_point)
        amari_chentsov_tensor = self.stat_metric.amari_chentsov_tensor(base_point)
        return LC_first_kind + ((1 + self.alpha) / 2) * amari_chentsov_tensor
    
    def christoffels(self, base_point):
        first_kind_christoffels = self.first_kind_christoffels(base_point)
        cometric = self.stat_metric.cometric_matrix(base_point)
        return gs.einsum('...lk, ...lij -> ...kij', cometric, first_kind_christoffels)