"""Varifolds related machinery.

General framework is introduced in [KCC2017]_.
See [CCGGR2020]_ for details about kernels.
Implementation is based in pykeops (https://www.kernel-operations.io/keops/).
In particular, see
https://www.kernel-operations.io/keops/_auto_tutorials/surface_registration/plot_LDDMM_Surface.html#data-attachment-term # noqa
for implementation details.

References
----------
.. [KCC2017] Irene Kaltenmark, Benjamin Charlier, and Nicolas Charon.
    “A General Framework for Curve and Surface Comparison and Registration
    With Oriented Varifolds,” 3346–55, 2017.
    https://openaccess.thecvf.com/content_cvpr_2017/html/Kaltenmark_A_General_Framework_CVPR_2017_paper.html.
.. [CCGGR2020] Nicolas Charon, Benjamin Charlier, Joan Glaunès, Pietro Gori, and Pierre Roussillon.
    “Fidelity Metrics between Curves and Surfaces: Currents, Varifolds, and Normal
    Cycles.” In Riemannian Geometric Statistics in Medical Image Analysis,
    edited by Xavier Pennec, Stefan Sommer, and Tom Fletcher, 441–77.
    Academic Press, 2020. https://doi.org/10.1016/B978-0-12-814725-2.00021-2
"""

import abc
import logging

import geomstats.backend as gs
from geomstats._mesh import Surface

from .kernel import GaussianBinetPairing

if gs.__name__.endswith("pytorch"):
    import torch

    def _gpu_is_available():
        return torch.cuda.is_available()

    def _to_device(array, device="cuda"):
        # TODO: check autodiff
        return array.to(device)


else:

    def _gpu_is_available():
        return False

    def _to_device(array, *args, **kwargs):
        return array


class KernelInducedMetric(abc.ABC):
    """Metric induced by a kernel pairing.

    This class represents metrics defined through a bilinear pairing
    induced by a kernel.

    Parameters
    ----------
    pairing : Pairing
        Object implementing the kernel pairing. It must provide methods
        to evaluate the kernel and associated reductions.
    """

    def __init__(self, pairing):
        self.pairing = pairing

    @abc.abstractmethod
    def transform(self, point):
        """Extract geometric features used by the representation."""

    def scalar_product(self, point_a, point_b):
        """Scalar product.

        Parameters
        ----------
        point_a : Surface
            A point.
        point_b : Surface
            A point.

        Returns
        -------
        scalar : float
        """
        point_a = self.transform(point_a)
        point_b = self.transform(point_b)

        return self.pairing(point_a, point_b)

    def squared_dist(self, point_a, point_b):
        """Squared distance.

        Parameters
        ----------
        point_a : Surface
            A point.
        point_b : Surface
            A point.

        Returns
        -------
        scalar : float
        """
        point_a = self.transform(point_a)
        point_b = self.transform(point_b)

        return (
            self.pairing(point_a, point_a)
            - 2 * self.pairing(point_a, point_b)
            + self.pairing(point_b, point_b)
        )

    def dist(self, point_a, point_b):
        """Squared distance.

        Parameters
        ----------
        point_a : Surface
            A point.
        point_b : Surface
            A point.

        Returns
        -------
        scalar : float
        """
        sq_dist = self.squared_dist(point_a, point_b)
        return gs.sqrt(sq_dist)

    def loss(self, target_point, target_faces=None):
        """Loss with respected to target point.

        Parameters
        ----------
        point_a : Surface
            A point.
        target_faces : array-like, shape=[n_faces, 3]
            Combinatorial structure of target mesh.

        Returns
        -------
        squared_dist : callable
            ``f(vertices) -> scalar``. Measures squared varifold distance
            between a point with ``vertices`` given wrt ``target_faces``
            against ``target_point``.
        """
        if target_faces is None:
            target_faces = target_point.faces

        target_point = self.transform(target_point)
        kernel_target = self.pairing(target_point, target_point)

        def squared_dist(vertices):
            point = Surface(vertices, target_faces)
            point = self.transform(point)
            return (
                kernel_target
                - 2 * self.pairing(target_point, point)
                + self.pairing(point, point)
            )

        return squared_dist


class VarifoldMetric(KernelInducedMetric):
    """Varifold metric.

    Parameters
    ----------
    sigma : float
        Positive bandwidth parameter of the Gaussian kernel.
    backend : {"auto", "torch", "keops", "keops_genred", "keops_lazy"}
        Implementation backend. Suffix with '_gpu' to control device.

        - "auto": Select an implementation automatically (prefers
          a KeOps-based implementation when available, otherwise falls back
          to a Torch/NumPy implementation).
        - "backend": Dense implementation using the current geomstats backend.
        - "keops": Alias for "keops_genred".
        - "keops_genred": KeOps implementation using Genred reductions.
        - "keops_lazy": KeOps LazyTensor-based implementation.
    """

    def __init__(self, sigma, backend="auto"):
        self.sigma = sigma
        pairing = GaussianBinetPairing(sigma, backend=backend)
        super().__init__(pairing)

        self._gpu = False
        if _gpu_is_available() and (backend == "auto" or backend.endswith("_gpu")):
            self._gpu = True

        if backend.endswith("_gpu") and not _gpu_is_available():
            logging.info("No GPU available, computing on CPU.")

    def transform(self, point):
        """Extract geometric features used by the varifold representation.

        Parameters
        ----------
        point : Surface
            Surface-like object with attributes ``face_centroids``,
            ``face_normals``, and ``face_areas``.

        Returns
        -------
        tuple
            Tuple ``(centroids, normals, areas)`` used in the kernel pairing.
        """
        arrays = (
            point.face_centroids,
            point.face_normals,
            point.face_areas,
        )

        if not self._gpu:
            return arrays

        return [_to_device(array) for array in arrays]
