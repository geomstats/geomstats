"""Kernel pairings."""

import importlib

import geomstats.backend as gs

from .base import Pairing

if gs.__name__.endswith("pytorch"):
    import torch

    _compile = torch.compile

else:

    def _compile(fn):
        return fn


def GaussianBinetPairing(sigma, backend="auto"):
    r"""Instantiate a Gaussian-Binet kernel pairing.

    This pairing is defined by

    .. math::

        K(x, y, u, v) = exp(-||x - y||^2 / sigma^2) * <u, v>^2


    Parameters
    ----------
    sigma : float
        Positive bandwidth parameter of the Gaussian kernel.
    backend : {"auto", "torch", "keops", "keops_genred", "keops_lazy"}
        Implementation backend.

        - "auto": Select an implementation automatically (typically prefers
          a KeOps-based implementation when available, otherwise falls back
          to a Torch/NumPy implementation).
        - "backend": Dense implementation using the current geomstats backend.
        - "keops": Alias for "keops_genred".
        - "keops_genred": KeOps implementation using Genred reductions.
        - "keops_lazy": KeOps LazyTensor-based implementation.

    Returns
    -------
    Pairing
        An object implementing the kernel pairing.

    Notes
    -----
    The dense ("backend") implementation materializes pairwise matrices and is
    memory-bound for large inputs. KeOps-based implementations avoid forming
    the full kernel matrix and are more efficient for large-scale problems.

    The "auto" backend does not guarantee optimal performance in all cases,
    but provides a reasonable default based on available dependencies.
    """
    if backend == "auto":
        has_keops = importlib.util.find_spec("pykeops") is not None
        backend = "keops_genred" if has_keops else "backend"

    if backend == "keops":
        backend = "keops_genred"

    if backend == "backend":
        return _GaussianBinetPairing(sigma=sigma)

    if backend == "keops_genred":
        import geomstats.varifold.keops.genred as gkeops

        return gkeops.GaussianBinetPairing(sigma)

    if backend == "keops_lazy":
        import geomstats.varifold.keops.lazy as lkeops

        return lkeops.SurfaceKernelPairing(lkeops.GaussianBinetKernel(sigma=sigma))

    raise ValueError(f"Unknown backend: {backend}")


class _GaussianBinetPairing(Pairing):
    r"""Instantiate a Gaussian–Binet kernel pairing.

    This pairing is defined by

    .. math::

        K(x, y, u, v) = exp(-||x - y||^2 / sigma^2) <u, v>^2

    Parameters
    ----------
    sigma : float
        Positive bandwidth parameter of the Gaussian kernel.

    Notes
    -----
    It materializes pairwise matrices and is memory-bound for large inputs.
    """

    def __init__(self, sigma):
        super().__init__()

        def _kernel(x, y, u, v):
            x_norm2 = gs.sum(x**2, axis=1)[:, None]
            y_norm2 = gs.sum(y**2, axis=1)[None, :]

            dist2 = x_norm2 + y_norm2 - 2 * (x @ y.T)
            K_xy = gs.exp(-dist2 / sigma**2)

            uv = u @ v.T

            return K_xy * uv**2

        self._kernel = _compile(_kernel)

    def kernel_prod(self, x, y, u, v, b):
        """Apply the kernel pairing to a vector."""
        return self._kernel(x, y, u, v) @ b
