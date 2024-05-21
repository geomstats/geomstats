import geomstats.backend as gs

if gs.__name__.endswith("numpy"):
    from pykeops.numpy import Vi, Vj
else:
    from pykeops.torch import Vi, Vj


# https://www.kernel-operations.io/keops/_auto_tutorials/surface_registration/plot_LDDMM_Surface.html#data-attachment-term

# TODO: delete?
SPATIAL_KERNEL = {
    "gaussian": "Exp(-SqDist(x,y)*a)",
    "cauchy": "IntCst(1)/(IntCst(1)+SqDist(x,y)*a)",
    "energy": "-Clamp(Norm2(x-y),cmin,cmax)",
}

GRASSMANIAN_KERNEL = {
    "constant": "IntCst(1)",
    "linear": "(u|v)",
    "gaussian_oriented": "Exp(IntCst(2)*b*((u|v)-IntCst(1)))",
    "binet": "Square((u|v))",
    "gaussian_unoriented": "Exp(IntCst(2)*b*(Square((u|v))-IntCst(1)))",
}

SIGNAL_KERNEL = {
    "constant": "IntCst(1)",
    "gaussian": " Exp(-SqDist(g,h)*c)",
    "cauchy": "IntCst(1)/(IntCst(1)+SqDist(g,h)*c)",
}


class Surface:
    def __init__(self, vertices, faces, signal=None):
        self.vertices = vertices
        self.faces = faces
        self.signal = signal

        (self.face_centroids, self.face_normals, self.face_areas) = (
            self._compute_mesh_info()
        )

    def _compute_mesh_info(self):
        slc = tuple([slice(None)] * len(self.vertices.shape[:-2]))
        face_coordinates = self.vertices[*slc, self.faces]
        vertex_0, vertex_1, vertex_2 = (
            face_coordinates[*slc, :, 0],
            face_coordinates[*slc, :, 1],
            face_coordinates[*slc, :, 2],
        )

        face_centroids = (vertex_0 + vertex_1 + vertex_2) / 3
        normals = 0.5 * gs.cross(vertex_1 - vertex_0, vertex_2 - vertex_0)
        # TODO: may need to expand dims
        area = gs.linalg.norm(normals, axis=-1)
        unit_normals = gs.einsum("...ij,...i->...ij", normals, 1 / area)

        return face_centroids, unit_normals, gs.expand_dims(area, axis=-1)


class SurfacesKernel:
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
        reduction_inputs = ()
        if self._has_position:
            reduction_inputs += (point_a.face_centroids, point_b.face_centroids)

        if self._has_tangent:
            reduction_inputs += (point_a.face_normals, point_b.face_normals)

        if self._has_signal:
            reduction_inputs += (point_a.signal, point_b.signal)

        reduction_inputs += (point_b.face_areas,)
        return gs.sum(self._kernel(*reduction_inputs) * point_a.face_areas)


def GaussianKernel(sigma, init_index=0, dim=3):
    r"""Implements Gaussian kernel.

    .. math ::

        K(x, y)=e^{-\|x-y\|^2 / \sigma^2}

    Parameters
    ----------
    sigma : float
    """
    # Exp(-SqDist(x,y)*a)
    x, y = Vi(init_index, dim), Vj(init_index + 1, dim)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    return (-D2 * gamma).exp()


def LinearKernel(init_index=0, dim=3):
    # (u|v)
    u, v = Vi(init_index, dim), Vj(init_index + 1, dim)
    return (u * v).sum() ** 2


def BinetKernel(init_index=0, dim=3):
    # Square((u|v))
    u, v = Vi(init_index, dim), Vj(init_index + 1, dim)
    return (u * v).sum() ** 2


def UnorientedGaussian(b=1.0, init_index=0, dim=3):
    # TODO: review name
    # Exp(IntCst(2)*b*((u|v)-IntCst(1)))
    u, v = Vi(init_index, dim), Vj(init_index + 1, dim)
    return (2 * b * ((u * v).sum() - 1)).exp()


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
        return self.kernel(point_a, point_b)

    def squared_dist(self, point_a, point_b):
        return (
            self.kernel(point_a, point_a)
            - 2 * self.kernel(point_a, point_b)
            + self.kernel(point_b, point_b)
        )

    def loss(self, target_point, target_faces=None):
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
