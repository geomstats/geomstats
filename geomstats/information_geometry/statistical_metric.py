"""Statistical metric and conjugate connections to equip manifold."""

import geomstats.backend as gs
from geomstats.geometry.connection import Connection
from geomstats.geometry.riemannian_metric import RiemannianMetric


class DivergenceConjugateConnection(Connection):
    r"""Class to derive the pair of conjugate connections from a divergence.

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
        assert space.intrinsic, (
            "The manifold must be parametrized by an intrinsic coordinate system."
        )
        self.dim = space.dim
        self.divergence = self._unpack_inputs(divergence)

    def _unpack_inputs(self, func):
        r"""Wrap function to unpack inputs to divergence function.

        Compose the divergence function :math:`D` with a wrapper :math:`f`
        to identify
        .. math::
            \tidle{D} = f\circ D: \mathbb{R}^{2n} \to \mathbb{R}
            D: \mathbb{R}^{n} \times \mathbb{R}^{n} \to \mathbb{R}

        Autodifferentiation is then computed on :math:`\tidle{D}`.

        Parameters
        ----------
        func : callable
            Function to unpack inputs.

        Returns
        -------
        callable
            Composition of wrapper function after divergence.
        """
        def wrapper(tensor):
            return func(tensor[..., : self.dim], tensor[..., self.dim :])
        return wrapper

    def divergence_christoffels(self, base_point):
        r"""Compute the Christoffel symbols of the divergence connection.

        Compute the Christoffel symbols of the divergence connection :math:`\nabla^D`
        at the tangent space of the base point.

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
            Christoffel symbols of the divergence connection.
        """
        hess = gs.autodiff.hessian(self.divergence)
        jac_hess = gs.autodiff.jacobian(hess)
        base_point_pair = gs.concatenate([base_point, base_point])
        return -1 * jac_hess(base_point_pair)[: self.dim, : self.dim, self.dim :]

    def dual_divergence_christoffels(self, base_point):
        r"""Compute the Christoffel symbols of the dual divergence connection.

        Compute the Christoffel symbols of the dual divergence connection
        :math:`\nabla^{D^*}` at the tangent space of the base point.

        .. math::
            \Gamma^{D^*}_{i j k} =
            -1 \cdot \frac{\partial^2}{\partial x^i}
                \frac{\partial^2}{\partial y^j \partial y^k} D(x, y) \bigg|_{x=y}

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim, dim]
            Christoffel symbols of the dual divergence connection.
        """
        hess = gs.autodiff.hessian(self.divergence)
        jac_hess = gs.autodiff.jacobian(hess)
        base_point_pair = gs.concatenate([base_point, base_point])
        return -1 * jac_hess(base_point_pair)[: self.dim, self.dim :, self.dim :]


class StatisticalMetric(RiemannianMetric):
    r"""Class to define a divergence induced statistical metric on a manifold.

    Given an :math:`n` manifold :math:`M` with coordinates :math:`x` and
    a divergence :math:`D: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}`,
    :math:`M` can be equipped with a divergence induced  metric :math:`g^D`
    and  a (0, 3) tensor :math:`C^D` called the Amari divergence tensor.

    The corresponding statistical manifold is therefore
    .. math::
        (M, g^D, C^D)

    Attributes
    ----------
    space : Manifold
        Manifold to equip with the divergence induced metric.
    divergence : callable
        Divergence function, takes two inputs of shape=[..., dim] and returns a scalar.

    References
    ----------
    .. [N2020] F. Nielsen,
        "An Elementary Introduction to Information Geometry",
        arXiv:808.08271v2 (2020): 10-16
    """

    def __init__(self, space, divergence):
        super().__init__(space=space, signature=(space.dim, 0))
        assert space.intrinsic, (
            "The manifold must be parametrized by an intrinsic coordinate system."
        )
        self.divergence_conjugate_connection = DivergenceConjugateConnection(
            space=space, divergence=divergence
        )

        self.dim = space.dim
        self.divergence = self._unpack_inputs(divergence)

    def __getattr__(self, attr):
        r"""Built in method to delegate attribute access.

        Delegate attribute access to the divergence conjugate connection.

        Instanciated to avoid dimond inheritance problem with RiemannianMetric
        and DivergenceConjugateConnection classes.

        Parameters
        ----------
        attr : str
            Attribute to access.

        Returns
        -------
        callable
            Attribute of divergence conjugate connection.
        """
        return getattr(self.divergence_conjugate_connection, attr)

    def _unpack_inputs(self, func):
        r"""Wrap function to unpack inputs to divergence function.

        Compose the divergence function :math:`D` with a wrapper :math:`f`
        to identify
        .. math::
            \tidle{D} = f\circ D: \mathbb{R}^{2n} \to \mathbb{R}
            D: \mathbb{R}^{n} \times \mathbb{R}^{n} \to \mathbb{R}

        Autodifferentiation is then computed on :math:`\tidle{D}`.

        Parameters
        ----------
        func : callable
            Function to unpack inputs.

        Returns
        -------
        callable
            Composition of wrapper function after divergence.
        """
        def wrapper(tensor):
            return func(tensor[..., : self.dim], tensor[..., self.dim :])
        return wrapper

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
        divergence_christoffels = self.divergence_christoffels(base_point)
        dual_divergence_christoffels = self.dual_divergence_christoffels(base_point)
        return dual_divergence_christoffels - divergence_christoffels
