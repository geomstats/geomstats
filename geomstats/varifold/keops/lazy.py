"""Kernels and kernel pairings using lazy KeOps."""

import geomstats.backend as gs
from geomstats.varifold.base import Pairing

if gs.__name__.endswith("pytorch"):
    from pykeops.torch import Vi, Vj
else:
    from pykeops.numpy import Vi, Vj


class SurfaceKernelPairing(Pairing):
    """A kernel pairing on surfaces.

    Parameters
    ----------
    position_kernel : pykeops.LazyTensor
    tangent_kernel : pykeops.LazyTensor
    signal_kernel : pykeops.LazyTensor
    """

    def __init__(self, kernel):
        area_b = Vj(kernel.new_variable_index(), 1)
        self.kernel_prod = (kernel * area_b).sum_reduction(axis=1)


def GaussianKernel(sigma=1.0, init_index=0, dim=3):
    r"""Gaussian kernel.

    .. math::

        K(x, y)=e^{-\|x-y\|^2 / \sigma^2}

    Generates the expression: `Exp(-SqDist(x,y)*a)`.

    Parameters
    ----------
    sigma : float
        Kernel parameter.
    init_index : int
        Index of first symbolic variable.
    dim : int
        Ambient dimension.
    """
    x, y = Vi(init_index, dim), Vj(init_index + 1, dim)
    gamma = 1 / (sigma * sigma)
    sdist = x.sqdist(y)
    return (-sdist * gamma).exp()


def CauchyKernel(sigma=1.0, init_index=0, dim=3):
    r"""Cauchy kernel.

    .. math::

        K(x, y)=\frac{1}{1+\|x-y\|^2 / \sigma^2}

    Generates the expression: `IntCst(1)/(IntCst(1)+SqDist(x,y)*a)`.

    Parameters
    ----------
    sigma : float
        Kernel parameter.
    init_index : int
        Index of first symbolic variable.
    dim : int
        Ambient dimension.
    """
    x, y = Vi(init_index, dim), Vj(init_index + 1, dim)
    gamma = 1 / (sigma * sigma)
    sdist = x.sqdist(y)
    return 1 / (1 + sdist * gamma)


def LinearKernel(init_index=0, dim=3):
    r"""Linear kernel.

    .. math::

        K(u, v) = \langle u, v \rangle

    Generates the expression: `(u|v)`.

    Parameters
    ----------
    init_index : int
        Index of first symbolic variable.
    dim : int
        Ambient dimension.
    """
    u, v = Vi(init_index, dim), Vj(init_index + 1, dim)
    return (u * v).sum()


def BinetKernel(init_index=0, dim=3):
    r"""Binet kernel.

    .. math::

        K(u, v) = \langle u, v \rangle^2

    Generates the expression: `Square((u|v))`.

    Parameters
    ----------
    init_index : int
        Index of first symbolic variable.
    dim : int
        Ambient dimension.
    """
    u, v = Vi(init_index, dim), Vj(init_index + 1, dim)
    return (u * v).sum() ** 2


def RestrictedGaussianKernel(sigma=1.0, oriented=False, init_index=0, dim=3):
    r"""Gaussian kernel restricted to the hypersphere.

    If unoriented:

    .. math::

        K(u, v)=e^{2 (\langle u, v \rangle ^2 - 1) / \sigma^2 }

    If oriented:

    .. math::

        K(u, v)=e^{2 (\langle u, v \rangle / - 1) / \sigma^2}

    Generates the expression:

    * oriented: `Exp(IntCst(2)*a*((u|v)-IntCst(1)))`
    * unoriented: `Exp(IntCst(2)*a*(Square((u|v))-IntCst(1)))`

    Parameters
    ----------
    sigma : float
        Kernel parameter.
    oriented : bool
        If False, uses squared inner product.
    init_index : int
        Index of first symbolic variable.
    dim : int
        Ambient dimension.
    """
    u, v = Vi(init_index, dim), Vj(init_index + 1, dim)
    b = 1 / (sigma * sigma)

    inner = (u * v).sum()
    if not oriented:
        inner = inner**2
    return (2 * b * (inner - 1)).exp()


def GaussianBinetKernel(sigma=1.0, init_index=0, dim=3):
    r"""Gaussian-Binet kernel.

    .. math::

        K(x, y, u, v)=e^{-\|x-y\|^2 / \sigma^2} \langle u, v \rangle^2

    Generates the expression: `Exp(-SqDist(x,y)*a)*Square((u|v))`.

    Parameters
    ----------
    sigma : float
        Kernel parameter.
    init_index : int
        Index of first symbolic variable.
    dim : int
        Ambient dimension.
    """
    position_kernel = GaussianKernel(sigma=sigma, init_index=init_index, dim=dim)
    tangent_kernel = BinetKernel(
        init_index=position_kernel.new_variable_index(), dim=dim
    )

    return position_kernel * tangent_kernel
