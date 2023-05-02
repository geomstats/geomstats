"""Discrete Surfaces with Elastic metrics.

Lead authors: Emmanuel Hartman, Adele Myers.
"""

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.manifold import Manifold


class DiscreteSurfaces(Manifold):
    r"""Space of parameterized discrete surfaces.

    Each surface is sampled with fixed n_vertices vertices and n_faces faces
    in $\mathbb{R}^3$.

    Each individual surface is represented by a 2d-array of shape `[
    n_vertices, 3]`. This space corresponds to the space of immersions
    defined below, i.e. the
    space of smooth functions from a template to manifold $M$ into  $\mathbb{R}^3$,
    with non-vanishing Jacobian.
    .. math::
        Imm(M,\mathbb{R}^3)=\{ f \in C^{\infty}(M, \mathbb{R}^3)
        \|Df(x)\|\neq 0 \forall x \in M \}.

    Parameters
    ----------
    faces : integer array-like, shape=[n_faces, 3]

    Attributes
    ----------
    faces : integer array-like, shape=[n_faces, 3]
    """

    def __init__(self, faces, **kwargs):
        """Create an object."""
        ambient_dim = 3
        self.faces = faces
        self.n_faces = len(faces)
        self.n_vertices = int(gs.amax(self.faces) + 1)
        dim = self.n_vertices * ambient_dim
        self.shape = (self.n_vertices, ambient_dim)
        super().__init__(
            dim=dim,
            shape=(self.n_vertices, 3),
            **kwargs,
        )

    def belongs(self, point, atol=gs.atol):
        """Test whether a point belongs to the manifold.

        Checks that vertices are inputed in proper form and are
        consistent with the mesh structure.

        Parameters
        ----------
        point : array-like, shape=[..., n_vertices, 3]
            Surface, i.e. the vertices of its triangulation.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : bool
            Boolean evaluating if point belongs to the space of discrete
            surfaces.
        """
        n_points = 1 if point.ndim == 2 else point.shape[0]
        if point.shape[-1] != 3:
            return gs.array([False] * n_points)
        if point.shape[-2] != self.n_vertices:
            return gs.array([False] * n_points)
        return gs.array([True] * n_points)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Tangent vectors are identified with points of the vector space so
        this checks the shape of the input vector.

        Parameters
        ----------
        vector : array-like, shape=[..., n_vertices, 3]
            Vector.
        base_point : array-like, shape=[..., n_vertices, 3]
            Point in the vector space.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : array-like, shape=[...,]
            Boolean denoting if vector is a tangent vector at the base point.
        """
        return vector.shape[-1] == 3 and vector.shape[-2] == self.n_vertices

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.

        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        """
        return vector

    def projection(self, point):
        """Project a point to the manifold.

        Parameters
        ----------
        point: array-like, shape[..., *point_shape]
            Point.

        Returns
        -------
        point: array-like, shape[..., *point_shape]
            Point.
        """
        return point

    def random_point(self, n_samples=1):
        """Sample discrete surfaces.

        This sample random discrete surfaces with the correct number of vertices.

        Parameters
        ----------
        n_samples : int
            Number of surfaces to sample.
            Optional, Default=1

        Returns
        -------
        vertices :  array-like, shape=[n_samples, n_vertices, 3]
            Vertices for a batch of points in the space of discrete surfaces.
        """
        euclidean = Euclidean(dim=3)
        vertices = euclidean.random_point(n_samples * self.n_vertices)
        vertices = gs.reshape(vertices, (n_samples, self.n_vertices, 3))
        return vertices[0] if n_samples == 1 else vertices

    def vertex_areas(self, point):
        """Compute vertex areas for a triangulated surface.

        Heron's formula gives the triangle's area in terms of its sides a b c:,
        As the square root of the product s(s - a)(s - b)(s - c),
        where s is the semiperimeter of the triangle, that is, s = (a + b + c)/2.

        The vertex area is defined as 2/3 of the sum of the areas of the faces
        touching it.

        Parameters
        ----------
        point : array-like, shape=[n_vertices, 3]
             Surface, i.e. the vertices of its triangulation.

        Returns
        -------
        vertareas :  array-like, shape=[n_vertices, 1]
            vertex areas
        """
        n_vertices, _ = point.shape
        face_coordinates = point[self.faces]
        vertex0, vertex1, vertex2 = (
            face_coordinates[:, 0],
            face_coordinates[:, 1],
            face_coordinates[:, 2],
        )
        len_edge_12 = gs.linalg.norm((vertex1 - vertex2), axis=1)
        len_edge_02 = gs.linalg.norm((vertex0 - vertex2), axis=1)
        len_edge_01 = gs.linalg.norm((vertex0 - vertex1), axis=1)
        half_perimeter = 0.5 * (len_edge_12 + len_edge_02 + len_edge_01)
        area = gs.sqrt(
            (
                half_perimeter
                * (half_perimeter - len_edge_12)
                * (half_perimeter - len_edge_02)
                * (half_perimeter - len_edge_01)
            ).clip(min=1e-6)
        )

        id_vertices = gs.flatten(self.faces)

        val = gs.flatten(gs.stack([area] * 3, axis=1))

        incident_areas = gs.zeros(n_vertices)
        incident_areas = gs.scatter_add(
            incident_areas, dim=0, index=id_vertices, src=val
        )
        vertex_areas = 2 * incident_areas / 3.0
        return vertex_areas

    def normals(self, point):
        """Compute normals at each face of a triangulated surface.

        Normals are the cross products between edges of each face
        that are incident to its x-coordinate.

        Parameters
        ----------
        point : array-like, shape=[n_vertices, 3]
            Surface, i.e. the vertices of its triangulation.

        Returns
        -------
        normals_at_point : array-like, shape=[n_facesx1]
            Normals of each face of the mesh.
        """
        vertex_0, vertex_1, vertex_2 = (
            gs.take(point, indices=self.faces[:, 0], axis=0),
            gs.take(point, indices=self.faces[:, 1], axis=0),
            gs.take(point, indices=self.faces[:, 2], axis=0),
        )
        normals_at_point = 0.5 * gs.cross(vertex_1 - vertex_0, vertex_2 - vertex_0)
        return normals_at_point

    def surface_one_forms(self, point):
        """Compute the vector valued one-forms.

        The one forms are evaluated at the faces of a triangulated surface.

        A one-form is represented by the two vectors (01) and (02) at each face
        of vertices 0, 1, 2.

        Parameters
        ----------
        point : array-like, shape=[n_vertices, 3]
             One surface, i.e. the vertices of its triangulation.

        Returns
        -------
        one_forms_base_point : array-like, shape=[n_faces, 3, 2]
            One form evaluated at each face of the triangulated surface.
        """
        vertex_0 = gs.take(point, indices=self.faces[:, 0], axis=-2)
        vertex_1 = gs.take(point, indices=self.faces[:, 1], axis=-2)
        vertex_2 = gs.take(point, indices=self.faces[:, 2], axis=-2)
        return gs.stack([vertex_1 - vertex_0, vertex_2 - vertex_0], axis=1)
        # point = gs.array(point)
        # need_squeeze = False
        # if point.ndim == 2:
        #     need_squeeze = True
        #     point = gs.expand_dims(point, axis=0)

        # vertex_0, vertex_1, vertex_2 = (
        #     gs.take(point, indices=self.faces[:, 0], axis=-2),
        #     gs.take(point, indices=self.faces[:, 1], axis=-2),
        #     gs.take(point, indices=self.faces[:, 2], axis=-2),
        # )

        # if need_squeeze:
        #     return gs.squeeze(
        #         gs.stack([vertex_1 - vertex_0, vertex_2 - vertex_0], axis=-1), axis=0
        #     )
        # return gs.stack([vertex_1 - vertex_0, vertex_2 - vertex_0], axis=-1)

    def face_areas(self, point):
        """Compute the areas for each face of a triangulated surface.

        The corresponds to the volume area for the surface metric, that is
        the volume area of the pullback metric of the immersion defining the
        surface metric.

        Parameters
        ----------
        point :  array-like, shape=[n_vertices, 3]
            One surface, i.e. the vertices of its triangulation.

        Returns
        -------
        _ :  array-like, shape=[n_faces,]
            Area computed at each face of the triangulated surface.
        """
        surface_metrics = self.surface_metric_matrices(point)
        return gs.sqrt(gs.linalg.det(surface_metrics))

    def surface_metric_matrices(self, point):
        """Compute the surface metric matrices.

        The matrices are evaluated at the faces of a triangulated surface.

        The surface metric is the pullback metric of the immersion q
        defining the surface, i.e. of
        the map q: M -> R3, where M is the parameterization manifold.

        Parameters
        ----------
        point : array like, shape=[n_vertices, 3]
            One surface, i.e. the vertices of its triangulation.

        Returns
        -------
        _ : array-like, shape=[n_faces, 2, 2]
            Surface metric matrices evaluated at each face of
            the triangulated surface.
        """
        one_forms = self.surface_one_forms(point)
        print("one_forms.shape", one_forms.shape)
        transposed_one_forms = gs.transpose(one_forms, axes=(0, 2, 1))
        return gs.matmul(one_forms, transposed_one_forms)
