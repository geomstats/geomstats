"""Unit tests for discrete_surfaces modules."""

import geomstats.backend as gs
from tests.conftest import Parametrizer, autograd_backend, pytorch_backend
from tests.data.discrete_surfaces_data import (
    DiscreteSurfacesTestData,
    ElasticMetricTestData,
)
from tests.geometry_test_cases import ManifoldTestCase, RiemannianMetricTestCase


class TestDiscreteSurfaces(ManifoldTestCase, metaclass=Parametrizer):

    testing_data = DiscreteSurfacesTestData()

    def test_vertex_areas(self, faces, point):
        """Test vertex_areas.

        Vertex area is the area of all of the triangles who are in contact
        with a specific vertex, according to the formula:
        vertex_areas = 2 * sum_incident_areas / 3.0

        We test this on a space whose initializing
        point is a cube, and we test the function on
        a cube with sides of length 2 centered at the origin.

        The cube is meshed with triangles, so each face should
        have area 2.
        """
        number_of_contact_faces = gs.array([3, 5, 5, 5, 5, 5, 3, 5])
        triangle_area = 0.5 * 2 * 2
        expected = 2 * (number_of_contact_faces * triangle_area) / 3
        space = self.Space(faces)

        result = space.vertex_areas(point)
        assert result.shape == (8,)
        assert expected.shape == (8,)
        assert gs.allclose(result, expected), result

        point = gs.array([point, point])
        expected = gs.array([expected, expected])
        result = space.vertex_areas(point)
        assert point.shape == (2, 8, 3)
        assert result.shape == (2, 8), result.shape
        assert gs.allclose(result, expected), result

    def test_normals(self, faces, point):
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
        space = self.Space(faces=faces)
        cube_normals = gs.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 2.0],
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, -2.0, 0.0],
                [-2.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, 0.0, -2.0],
                [0.0, 0.0, -2.0],
            ]
        )
        expected = cube_normals

        result = space.normals(point)
        are_close = [
            (gs.allclose(res, exp) or gs.allclose(res, -exp))
            for res, exp in zip(result, expected)
        ]

        assert gs.all(are_close)

        point = gs.array([point, point])
        result = space.normals(point)
        are_close_0 = [
            (gs.allclose(res, exp) or gs.allclose(res, -exp))
            for res, exp in zip(result[0], expected)
        ]
        are_close_1 = [
            (gs.allclose(res, exp) or gs.allclose(res, -exp))
            for res, exp in zip(result[1], expected)
        ]
        assert gs.all(gs.array([are_close_0, are_close_1]))

    def test_surface_one_forms(self, faces, point):
        """Test surface one forms."""
        space = self.Space(faces=faces)

        result = space.surface_one_forms(point=point)
        assert result.shape == (space.n_faces, 2, 3), result.shape

        first_vec = result[:, 0, :]
        second_vec = result[:, 1, :]
        inner_prods = gs.einsum("ni,ni->n", first_vec, second_vec)
        result = [prod in [0.0, 4.0] for prod in inner_prods]
        assert gs.all(result)

        singleton_point = gs.expand_dims(point, axis=0)
        result = space.surface_one_forms(point=singleton_point)
        assert result.shape == (1, space.n_faces, 2, 3)

        point = gs.array([point, point])
        result = space.surface_one_forms(point=point)
        assert result.shape == (2, space.n_faces, 2, 3)

        first_vec = result[:, :, 0, :]
        second_vec = result[:, :, 1, :]
        inner_prods = gs.einsum("mni,mni->mn", first_vec, second_vec)
        result = []
        for inner_prod in inner_prods:
            result.append([prod in [0.0, 4.0] for prod in inner_prod])
        assert gs.all(result)

    def test_faces_area(self, faces, point):
        """Test faces area."""
        space = self.Space(faces=faces)

        result = space.face_areas(point=point)
        expected = gs.array([4.0] * 12)
        assert result.shape == (space.n_faces,), result.shape
        assert gs.allclose(result, expected), result

        point = gs.array([point, point])
        result = space.face_areas(point=point)
        expected = gs.array([expected, expected])
        assert result.shape == (2, space.n_faces), result.shape
        assert gs.allclose(result, expected), result

    def test_surface_metric_matrices(self, faces, point):
        """Test surface metric matrices."""
        space = self.Space(faces=faces)
        result = space.surface_metric_matrices(point=point)
        assert result.shape == (
            space.n_faces,
            2,
            2,
        ), result.shape

        point = gs.array([point, point])
        result = space.surface_metric_matrices(point=point)
        assert result.shape == (2, space.n_faces, 2, 2)

    def test_laplacian(self, faces, point, tangent_vec, expected):
        """Test laplacian operator."""
        space = self.Space(faces=faces)

        n_vertices = point.shape[-2]
        result = space.laplacian(point=point)(tangent_vec)
        assert result.shape == (n_vertices, 3), result.shape

        assert gs.allclose(result, expected), result

        tangent_vec = gs.array([tangent_vec, tangent_vec])
        result = space.laplacian(point=point)(tangent_vec)
        assert result.shape == (2, n_vertices, 3), result.shape


class TestElasticMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_all = not (autograd_backend() or pytorch_backend())
    skip_test_exp_shape = True
    skip_test_log_shape = True
    skip_test_exp_geodesic_ivp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_after_log = True
    skip_test_exp_belongs = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_log_after_exp = True
    skip_test_log_is_tangent = True
    skip_test_dist_is_norm_of_log = True
    skip_test_dist_is_positive = True
    skip_test_dist_is_symmetric = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_triangle_inequality_of_dist = True
    skip_test_squared_dist_is_symmetric = True
    skip_test_squared_dist_is_positive = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = ElasticMetricTestData()

    def test_path_energy_per_time_is_positive(
        self, space, a0, a1, b1, c1, d1, a2, path, atol
    ):
        """Check that energy of a path of surfaces is positive at each time-step.

        Parameters
        ----------
        space : DiscreteSurfaces
            Space of discrete surfaces associated with the ElasticMetric.
        path : array-like, shape=[n_time_steps, n_vertices, 3]
            Path in the space of discrete surfaces.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, a0=a0, a1=a1, b1=b1, c1=c1, d1=d1, a2=a2)

        energy = space.metric.path_energy_per_time(path)
        result = gs.all(energy > -1 * atol)
        self.assertTrue(result)

    def test_path_energy_is_positive(self, space, a0, a1, b1, c1, d1, a2, path, atol):
        """Check that energy of a path of surfaces is positive at each time-step.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        path : array-like, shape=[n_time_steps, n_vertices, 3]
            Path in the space of discrete surfaces.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, a0=a0, a1=a1, b1=b1, c1=c1, d1=d1, a2=a2)

        energy = space.metric.path_energy(path)
        result = gs.all(energy > -1 * atol)
        self.assertTrue(result)
