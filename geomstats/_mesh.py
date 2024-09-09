"""Mesh-related data structures."""

import geomstats.backend as gs


class Surface:
    """A surface mesh.

    Mesh info (face centroids, normals and areas) is cached
    for greater performance.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Mesh vertices.
    faces : array-like, shape=[n_faces, 3]
        Mesh faces.
    signal : array-like, shape=[n_faces, d]
        A signal of the surface.
    """

    def __init__(self, vertices, faces, signal=None):
        self.vertices = vertices
        self.faces = faces
        self.signal = signal

        (self.face_centroids, self.face_normals, self.face_areas) = (
            self._compute_mesh_info()
        )

    def _compute_mesh_info(self):
        """Compute mesh information.

        Returns
        -------
        face_centroids : array-like, shape=[n_faces, 3]
            Face centroids.
        unit_normals : array-like, shape=[n_faces, 3]
            Face unit normals.
        face_areas : array-like, shape=[n_faces, 1]
            Face areas.
        """
        batch_slc = tuple([slice(None)] * len(self.vertices.shape[:-2]))
        face_coordinates = self.vertices[batch_slc + (self.faces,)]
        vertex_0, vertex_1, vertex_2 = (
            face_coordinates[batch_slc + (slice(None), 0)],
            face_coordinates[batch_slc + (slice(None), 1)],
            face_coordinates[batch_slc + (slice(None), 2)],
        )

        face_centroids = (vertex_0 + vertex_1 + vertex_2) / 3
        normals = 0.5 * gs.cross(vertex_1 - vertex_0, vertex_2 - vertex_0)
        area = gs.linalg.norm(normals, axis=-1)
        unit_normals = gs.einsum("...ij,...i->...ij", normals, 1 / area)

        return face_centroids, unit_normals, gs.expand_dims(area, axis=-1)
