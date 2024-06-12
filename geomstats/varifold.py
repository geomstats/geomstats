"""(Oriented) varifolds related machinery.

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

import geomstats.backend as gs

if gs.__name__.endswith("pytorch"):
    from pykeops.torch import Vi, Vj
else:
    from pykeops.numpy import Vi, Vj


from geomstats._mesh import Surface


class SurfacesKernel:
    """A kernel on surfaces.

    Parameters
    ----------
    position_kernel : pykeops.LazyTensor
    tangent_kernel : pykeops.LazyTensor
    signal_kernel : pykeops.LazyTensor
    """

    def __init__(
        self,
        position_kernel=None,
        tangent_kernel=None,
        signal_kernel=None,
    ):
        reduction = 1.0

        self._has_position = False
        if position_kernel is not None:
            self._has_position = True
            reduction *= position_kernel

        self._has_tangent = False
        if tangent_kernel is not None:
            self._has_tangent = True
            reduction *= tangent_kernel

        self._has_signal = False
        if signal_kernel is not None:
            self._has_signal = True
            reduction *= signal_kernel

        area_b = Vj(reduction.new_variable_index(), 1)
        self._kernel = (reduction * area_b).sum_reduction(axis=1)

    def __call__(self, point_a, point_b):
        """Evaluate kernel.

        Parameters
        ----------
        point_a : Surface
        point_b : Surface

        Returns
        -------
        scalar : float
        """
        reduction_inputs = ()
        if self._has_position:
            reduction_inputs += (point_a.face_centroids, point_b.face_centroids)

        if self._has_tangent:
            reduction_inputs += (point_a.face_normals, point_b.face_normals)

        if self._has_signal:
            reduction_inputs += (point_a.signal, point_b.signal)

        reduction_inputs += (point_b.face_areas,)
        return gs.sum(self._kernel(*reduction_inputs) * point_a.face_areas)


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

        K(u, v)=e^{2 (\langle u, v \rangle / - 1) - 1}

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


class VarifoldMetric:
    """Varifold metric.

    Parameters
    ----------
    kernel : callable
    """

    def __init__(self, kernel=None):
        if kernel is None:
            position_kernel = GaussianKernel(sigma=1.0, init_index=0)
            tangent_kernel = BinetKernel(
                init_index=position_kernel.new_variable_index()
            )

            kernel = SurfacesKernel(position_kernel, tangent_kernel)

        self.kernel = kernel

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
        return self.kernel(point_a, point_b)

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
        return (
            self.kernel(point_a, point_a)
            - 2 * self.kernel(point_a, point_b)
            + self.kernel(point_b, point_b)
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
            `f(vertices) -> scalar`. Measures squared varifold distance
            between a point with `vertices` given wrt `target_faces`
            against `target_point`.
        """
        if target_faces is None:
            target_faces = target_point.faces

        kernel_target = self.kernel(target_point, target_point)

        def squared_dist(vertices):
            point = Surface(vertices, target_faces)
            return (
                kernel_target
                - 2 * self.kernel(target_point, point)
                + self.kernel(point, point)
            )

        return squared_dist
