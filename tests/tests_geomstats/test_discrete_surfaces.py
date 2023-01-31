"""Unit tests for discrete_surfaces modules.

Add your code directory to the PYTHON PATH before running.
export PYTHONPATH=/home/nmiolane/code.

Get the .npy files to test this code.
"""

import os

import numpy as np
from my28brains.my28brains.discrete_surfaces import DiscreteSurfaces

import geomstats.backend as gs

# from geomstats.geometry.connection import Connection


# import sphere data vertices and faces
BRAINS_DIR = os.path.join(os.environ["PYTHONPATH"], "my28brains")
DATA_DIR = os.path.join(BRAINS_DIR, "data")

# uncomment if you would like to test a cube
CUBE_MESH_DIR = os.path.join(DATA_DIR, "cube_meshes")
test_vertices_path = os.path.join(CUBE_MESH_DIR, "vertices.npy")
test_faces_path = os.path.join(CUBE_MESH_DIR, "faces.npy")

test_vertices = np.load(test_vertices_path)
test_faces = np.load(test_faces_path)
test_vertices = gs.cast(test_vertices, gs.int64)
test_faces = gs.cast(test_faces, gs.int64)


# for testing a sphere -used for tangent testing
# SPHERE_DATA_DIR = os.path.join(DATA_DIR, "sphere_meshes")
# test_sphere_vertices_path = os.path.join(SPHERE_DATA_DIR, "vertices.npy")
# test_sphere_faces_path = os.path.join(SPHERE_DATA_DIR, "faces.npy")

# test_sphere_vertices = np.load(test_sphere_vertices_path)
# test_sphere_faces = np.load(test_sphere_faces_path)
# test_sphere_vertices = gs.cast(test_sphere_vertices, gs.int64)
# test_sphere_faces = gs.cast(test_sphere_faces, gs.int64)


def test_belongs():
    """Test that a set of vertices belongs to the manifold of DiscreteSurfaces.

    (Also inheritely checks if the discrete surface has degenerate triangles.)
    """
    vertices = test_vertices
    space = DiscreteSurfaces(faces=test_faces)
    space.belongs(point=vertices)


def test_is_tangent():
    """Test is_tangent.

    TODO.  how to test whether a vector is tangent
    - create a vector you know is tangent
    - is there a property with the connection?
    """

    # space = DiscreteSurfaces(faces=test_faces)
    # base_point = test_vertices
    # point_2 = space.random_point()
    # tangent_vector = Connection.log(point_2, base_point)
    # assert space.is_tangent(tangent_vector, base_point)

    # non_tangent_vector=gs.array([1,0,1])
    # vertices = test_sphere_vertices
    # space = DiscreteSurfaces(faces=test_sphere_faces)
    # base_point = vertices[3]
    # tangent_vector = gs.array([1,0,0])
    # non_tangent_vector=gs.array([1,0,1])
    # print(base_point)
    # print(tangent_vector)
    # print(non_tangent_vector)
    # assert space.is_tangent(tangent_vector,base_point) == True
    # assert space.is_tangent(non_tangent_vector,base_point) != True


def test_random_point_1():
    """Test random_point.

    This function tests that:
    - random_point() generates the correct number of samples (3)
    """
    # ambient_dim = 3
    # faces = gs.ones((12, ambient_dim))
    space = DiscreteSurfaces(faces=test_faces)
    # use a random point a test_vertices
    point = space.random_point(n_samples=3)
    assert point.shape[-1] == 3


def test_random_point_2():
    """Test random_point.

    This function tests that:
    - random_point() generates a point on the manifold

    TODO: this is failing. I think its failure has something to do with
    it not being implemented in a way that is compatible with
    the belongs() function???

    error: IndexError: index 1 is out of bounds for axis 0 with size 1
    """
    space = DiscreteSurfaces(faces=test_faces)
    point = space.random_point(n_samples=1)
    print("POINT", point.shape)
    # print(point.shape)
    # print(test_vertices)
    # print(test_faces)
    # print(space.n_vertices)
    # print(space.faces)
    assert space.belongs(point)


def test_vertex_areas():
    """Test vertex_areas.

    TODO. vertex_areas() might be referring to voronoi area.
    not sure though. tbd.

    We test this on a space whose initializing
    point is a cube, and we test the function on
    a cube with sides of length 2 centered at the origin.

    The cube is meshed with triangles, so each face should
    have area 2.
    """
    # space = DiscreteSurfaces(faces=test_faces)
    # point = test_vertices
    # areas = gs.array([[12],[12],[12],[12],[12],[12],[12],[12],[12],[12],[12],[12]])
    # print(areas.shape)
    # assert (areas[0] == space.vertex_areas(point)[0]).all()


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
    of the normal vectors to the cube positive.
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
