"""Unit tests for discrete_surfaces modules.

Add your code directory to the PYTHON PATH before running.
export PYTHONPATH=/home/nmiolane/code.

Get the .npy files to test this code.
"""

import os

import numpy as np

import geomstats.backend as gs
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces

TESTS_DIR = os.path.join(os.getcwd(), "tests")
test_vertices_path = os.path.join(TESTS_DIR, "test_vertices.npy")
test_faces_path = os.path.join(TESTS_DIR, "test_faces.npy")
test_vertices_source_path = os.path.join(TESTS_DIR, "test_vertices_source.npy")
test_faces_source_path = os.path.join(TESTS_DIR, "test_faces_source.npy")
test_vertices_target_path = os.path.join(TESTS_DIR, "test_vertices_target.npy")
test_faces_target_path = os.path.join(TESTS_DIR, "test_faces_target.npy")

test_vertices = np.load(test_vertices_source_path)
test_faces = np.load(test_faces_source_path)

print(test_vertices.shape)
print(test_faces.shape)


def test_random_point():
    """Test random point."""
    ambient_dim = 3
    faces = gs.ones((12, ambient_dim))
    space = DiscreteSurfaces(faces=faces)
    point = space.random_point(n_samples=3)
    assert point.shape[-1] == 3


def test_surface_one_forms():
    """Test surface one forms."""
    vertices = test_vertices
    space = DiscreteSurfaces(faces=test_faces)
    one_forms = space.surface_one_forms(point=vertices)
    assert one_forms.shape == (space.n_faces, 2, 3), one_forms.shape


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


def test_faces_area():
    """Test faces area."""
    vertices = test_vertices
    space = DiscreteSurfaces(faces=test_faces)
    face_areas = space.face_areas(point=vertices)
    assert face_areas.shape == (space.n_faces,), face_areas.shape


def test_belongs():
    """Test that a set of vertices belongs to the manifold of DiscreteSurfaces."""
    vertices = test_vertices
    space = DiscreteSurfaces(faces=test_faces)
    space.belongs(point=vertices)
    # This will not belong since the degenerate faces have not been removed.
