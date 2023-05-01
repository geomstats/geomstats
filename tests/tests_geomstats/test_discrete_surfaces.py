"""Unit tests for discrete_surfaces modules."""
# TODO: test laplacian function

import os

import numpy as np

import geomstats.backend as gs
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "geomstats", "datasets", "data")
CUBE_MESH_DIR = os.path.join(DATA_DIR, "cube_meshes")
test_vertices_path = os.path.join(CUBE_MESH_DIR, "vertices.npy")
test_faces_path = os.path.join(CUBE_MESH_DIR, "faces.npy")
print(test_vertices_path)
print(test_faces_path)

test_vertices = gs.array(np.load(test_vertices_path))
test_faces = gs.array(np.load(test_faces_path))
test_vertices = gs.cast(test_vertices, gs.float64)


def test_belongs():
    """Test that a set of vertices belongs to the manifold of DiscreteSurfaces.

    (Also inheritely checks if the discrete surface has degenerate triangles.)
    """
    space = DiscreteSurfaces(faces=test_faces)
    assert space.belongs(point=test_vertices)


def test_random_point_shape():
    """Test random_point.

    This function tests that random_point() generates
    the correct number of samples.

    Where: one sample = one discrete surface.
    """
    space = DiscreteSurfaces(faces=test_faces)
    point = space.random_point(n_samples=3)
    result, _, _ = point.shape
    assert result == 3


def test_random_point_and_belongs():
    """Test random_point.

    This function tests that random_point() generates a point
    on the manifold, i.e. generates a discrete surface.
    """
    space = DiscreteSurfaces(faces=test_faces)
    point = space.random_point(n_samples=1)
    assert space.belongs(point)


def test_vertex_areas():
    """Test vertex_areas.

    Vertex area is the area of all of the triangles who are in contact
    with a specific vertex, according to the formula:
    vertex_areas = 2 * incident_areas / 3.0

    We test this on a space whose initializing
    point is a cube, and we test the function on
    a cube with sides of length 2 centered at the origin.

    The cube is meshed with triangles, so each face should
    have area 2.

    TODO: add "scatter_add" to geomstats' backend.
    """
    space = DiscreteSurfaces(faces=test_faces)
    point = test_vertices
    n_vertices, _ = point.shape
    number_of_contact_faces = gs.array([3, 5, 5, 5, 5, 5, 3, 5])
    triangle_area = 0.5 * 2 * 2
    expected = 2 * (number_of_contact_faces * triangle_area) / 3
    result = space.vertex_areas(point)
    assert gs.allclose(result, expected), result


def test_normals():
    """Test normals.

    We test this on a space whose initializing
    point is a cube, and we test the function on
    a cube with sides of length 2 centered at the origin.
    The cube is meshed with 12 triangles (2 triangles
    per face.)

    Recall that the magnitude of each normal vector is equal to
    the area of the face it is normal to.

    We compare the abs value of each normal vector array because:
    note that the "normals" variable here calculates the normals
    as pointing out of the surface, but the way that normals()
    was constructed makes it so that the normal vector could be
    pointing into the surface or out of the surface, (so it could
    either be positive or negative). Because of this, we make all
    of the normal vectors to the cube positive for testing
    purposes.
    """
    space = DiscreteSurfaces(faces=test_faces)
    point = test_vertices
    cube_normals = np.array(
        [
            [0, 0, 2],
            [0, 0, 2],
            [0, 2, 0],
            [0, 2, 0],
            [2, 0, 0],
            [2, 0, 0],
            [0, -2, 0],
            [0, -2, 0],
            [-2, 0, 0],
            [-2, 0, 0],
            [0, 0, -2],
            [0, 0, -2],
        ]
    )
    abs_cube_normals = gs.abs(cube_normals)
    abs_int_normals = gs.cast(gs.abs(space.normals(point)), gs.int64)
    for i_vect, cube_vector in enumerate(abs_cube_normals):
        assert (cube_vector == abs_int_normals[i_vect]).all()


def test_surface_one_forms():
    """Test surface one forms."""
    vertices = test_vertices
    space = DiscreteSurfaces(faces=test_faces)
    one_forms = space.surface_one_forms(point=vertices)
    assert one_forms.shape == (space.n_faces, 2, 3), one_forms.shape


def test_faces_area():
    """Test faces area."""
    vertices = test_vertices
    space = DiscreteSurfaces(faces=test_faces)
    face_areas = space.face_areas(point=vertices)
    assert face_areas.shape == (space.n_faces,), face_areas.shape


def test_surface_metric_matrices():
    """Test surface metric matrices."""
    vertices = test_vertices
    space = DiscreteSurfaces(faces=test_faces)
    surface_metric_matrices = space.surface_metric_matrices(point=vertices)
    assert surface_metric_matrices.shape == (
        space.n_faces,
        2,
        2,
    ), surface_metric_matrices.shape
