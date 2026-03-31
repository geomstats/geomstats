"""Kernels and kernel pairings using KeOps genred."""

import geomstats.backend as gs
from geomstats.varifold.base import Pairing

if gs.__name__.endswith("pytorch"):
    from pykeops.torch import Genred
else:
    from pykeops.numpy import Genred


def GaussianKernel(sigma):
    r"""Gaussian kernel.

    .. math::

        K(x, y)=e^{-\|x-y\|^2 / \sigma^2}

    Parameters
    ----------
    sigma : float
        Kernel parameter.
    """
    expr = Genred(
        "Exp(-SqDist(x,y)*a)",
        [
            "a=Pm(1)",
            "x=Vi(3)",
            "y=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )
    a_param = 1 / gs.array([sigma]) ** 2

    def kernel_eval(point_a, point_b):
        return expr(a_param, point_a, point_b)

    return kernel_eval


def CauchyKernel(sigma):
    r"""Cauchy kernel.

    .. math::

        K(x, y)=\frac{1}{1+\|x-y\|^2 / \sigma^2}

    Parameters
    ----------
    sigma : float
        Kernel parameter.
    """
    expr = Genred(
        "IntCst(1)/(IntCst(1)+SqDist(x,y)*a)",
        [
            "a=Pm(1)",
            "x=Vi(3)",
            "y=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )
    a_param = 1 / gs.array([sigma]) ** 2

    def kernel_eval(point_a, point_b):
        return expr(a_param, point_a, point_b)

    return kernel_eval


def LinearKernel():
    r"""Linear kernel.

    .. math::

        K(u, v) = \langle u, v \rangle
    """
    expr = Genred(
        "(u|v)",
        [
            "u=Vi(3)",
            "v=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    def kernel_eval(point_a, point_b):
        return expr(point_a, point_b)

    return kernel_eval


def BinetKernel():
    r"""Binet kernel.

    .. math::

        K(u, v) = \langle u, v \rangle^2
    """
    expr = Genred(
        "Square((u|v))",
        [
            "u=Vi(3)",
            "v=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    def kernel_eval(point_a, point_b):
        return expr(point_a, point_b)

    return kernel_eval


def OrientedGaussianKernel(sigma=1.0):
    r"""Gaussian kernel restricted to the hypersphere.

    .. math::

        K(u, v)=e^{2 (\langle u, v \rangle / - 1) / \sigma^2}

    Parameters
    ----------
    sigma : float
        Kernel parameter.
    """
    expr = Genred(
        "Exp(IntCst(2)*b*((u|v)-IntCst(1)))",
        [
            "b=Pm(1)",
            "u=Vi(3)",
            "v=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    a_param = 1 / gs.array([sigma]) ** 2

    def kernel_eval(point_a, point_b):
        return expr(a_param, point_a, point_b)

    return kernel_eval


def UnorientedGaussianKernel(sigma=1.0):
    r"""Gaussian kernel restricted to the hypersphere.

    .. math::

        K(u, v)=e^{2 (\langle u, v \rangle ^2 - 1) / \sigma^2 }


    Parameters
    ----------
    sigma : float
        Kernel parameter.
    """
    expr = Genred(
        "Exp(IntCst(2)*b*(Square((u|v))-IntCst(1)))",
        [
            "b=Pm(1)",
            "u=Vi(3)",
            "v=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    a_param = 1 / gs.array([sigma]) ** 2

    def kernel_eval(point_a, point_b):
        return expr(a_param, point_a, point_b)

    return kernel_eval


class GaussianBinetPairing(Pairing):
    r"""Instantiate a Gaussian–Binet kernel pairing.

    This pairing is defined by

    .. math::

        K(x, y, u, v) = exp(-||x - y||^2 / sigma^2) <u, v>^2

    Parameters
    ----------
    sigma : float
        Positive bandwidth parameter of the Gaussian kernel.
    """

    def __init__(self, sigma):
        super().__init__()
        self._expr = Genred(
            "Exp(-SqDist(x,y)*a)*Square((u|v))*b",
            [
                "a=Pm(1)",
                "x=Vi(3)",
                "y=Vj(3)",
                "u=Vi(3)",
                "v=Vj(3)",
                "b=Vj(1)",
            ],
            reduction_op="Sum",
            axis=1,
        )
        self._a_param = 1 / gs.array([sigma]) ** 2

    def kernel_prod(self, *kernel_args):
        """Apply the kernel pairing to a vector."""
        return self._expr(self._a_param, *kernel_args)
